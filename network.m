function [net, history] = network(varargin)
%% Train a small DeepRx-style CNN
% Run: network
% Five master seeds control independent streams of NEW simulation seeds.
% Change the number of master seeds by changing the list:
%   network('TrainingSeeds',42:51,'MaxEpochs',30)
% Prepare inputs without Deep Learning Toolbox:
%   network('PrepareOnly',true)
% deeprx_sample.m packs inputs identically for training and testing.
% The requested filename shadows MATLAB's older network constructor.
% This code uses the modern dlnetwork class instead.

%% 1. Editable defaults and optional name/value overrides
p = inputParser; p.PartialMatching = false;
seedList = @(x) isnumeric(x) && isvector(x) && ~isempty(x) && ...
    isreal(x) && all(isfinite(x)) && all(x>=0 & x<=2^32-1 & x==fix(x));
positiveInteger = @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x>=1 && x==fix(x);
addParameter(p,'TrainingSeeds',42:46,seedList);       % FIVE master seeds.
addParameter(p,'ValidationSeeds',1001:1002,seedList); % Never used for gradients.
addParameter(p,'SamplesPerSeed',200,positiveInteger); % Fresh slots per master seed PER EPOCH.
addParameter(p,'ValidationSamplesPerSeed',50,positiveInteger); % Fixed held-out slots.
addParameter(p,'SNRValues',0:5:20,@(x) isnumeric(x) && isvector(x) && ~isempty(x) && isreal(x) && all(isfinite(x)));
addParameter(p,'DopplerValues',0:100:500,@(x) isnumeric(x) && isvector(x) && ~isempty(x) && isreal(x) && all(isfinite(x) & x>=0));
addParameter(p,'MaxEpochs',60,positiveInteger);       % Passes over the dataset.
addParameter(p,'MiniBatchSize',4,positiveInteger);    % Slots per gradient update.
addParameter(p,'LearningRate',1e-3,@(x) isnumeric(x) && isscalar(x) && isreal(x) && isfinite(x) && x>0);
addParameter(p,'InitializationSeed',7,@(x) seedList(x) && isscalar(x));
addParameter(p,'ExecutionEnvironment','auto',@(x) any(strcmp(x,{'auto','cpu','gpu'})));
addParameter(p,'PrepareOnly',false,@(x) islogical(x) && isscalar(x));
addParameter(p,'ShowPlots',true,@(x) islogical(x) && isscalar(x));
parse(p,varargin{:}); training = p.Results;
assert(numel(unique(training.TrainingSeeds))==numel(training.TrainingSeeds), ...
    'TrainingSeeds must not contain duplicates.');
assert(numel(unique(training.ValidationSeeds))==numel(training.ValidationSeeds), ...
    'ValidationSeeds must not contain duplicates.');
assert(isempty(intersect(training.TrainingSeeds,training.ValidationSeeds)), ...
    'Training and validation seeds must be different.');
if ~training.PrepareOnly
    assert(~isempty(which('dlnetwork')) && ~isempty(which('groupedConvolution2dLayer')), ...
        ['Training requires Deep Learning Toolbox. ' ...
        'Use network(''PrepareOnly'',true) to prepare the data only.']);
    verifyLossGradient();
end
folder = fileparts(mfilename('fullpath'));
baseCfg = simo_config('SNRdB',20,'MaximumDopplerShift',100, ...
    'ShowPlots',false,'RunChecks',false,'SaveData',false,'Verbose',false);

%% 2. Plan independent slots, then prepare the first epoch and validation set
% Do not reset the generator to the same five seeds for every condition:
% that would repeat only five bit patterns and encourage memorization.
% Each master seed drives a reproducible sequence of distinct simulation
% seeds. Each slot and each epoch now has fresh bits, fading, and noise.
% Defaults: 500 slots/epoch * 20 epochs = 10000 distinct training slots.
roots = [training.TrainingSeeds(:);training.ValidationSeeds(:)];
trainingPlan = makePlan(training.TrainingSeeds,training.SNRValues, ...
    training.DopplerValues,training.SamplesPerSeed,training.MaxEpochs,roots);
validationSamples = makePlan(training.ValidationSeeds,training.SNRValues, ...
    training.DopplerValues,training.ValidationSamplesPerSeed,1, ...
    [roots;trainingPlan.Seed]);
assert(isempty(intersect(trainingPlan.Seed,validationSamples.Seed)));
trainingSamples = trainingPlan(trainingPlan.Epoch==1,:);
[Z,bitLabels,bitMask] = makeDataset(baseCfg,trainingSamples);
[validationZ,validationLabels,validationMask] = makeDataset(baseCfg,validationSamples);
% Z dimensions: frequency x time x features x samples (312 x 14 x 10 x 500).
% Labels/masks: frequency x time x bits per symbol x samples (4 for 16QAM).
% network_data.mat stores the FIRST epoch; trainingPlan records every epoch.    
save(fullfile(folder,'network_data.mat'),'Z','bitLabels','bitMask', ...
    'validationZ','validationLabels','validationMask', ...
    'trainingSamples','trainingPlan','validationSamples','training','baseCfg','-v7.3');
fprintf('Prepared %d training slots and %d validation slots.\n', ...
    size(Z,4),size(validationZ,4));
net = []; history = table();
if training.PrepareOnly
    fprintf('Saved network_data.mat. No network was trained.\n');
    return;
end

%% 3. Build the 11-block CNN (paper Table I)
previousRng = rng;
rngCleanup = onCleanup(@() rng(previousRng));
rng(training.InitializationSeed,'twister');
net = buildCNN([size(Z,1),size(Z,2),size(Z,3)],size(bitLabels,3));
useGPU = ~strcmp(training.ExecutionEnvironment,'cpu') && canUseGPU;
assert(~strcmp(training.ExecutionEnvironment,'gpu') || useGPU, ...
    'GPU was requested but is unavailable. Select cpu or auto.');
if useGPU, net = transferNetwork(net,@gpuArray); end
fprintf('Training with GPU: %d. Positive output scores mean bit 1.\n',useGPU);

%% 4. Custom training loop: masked binary cross-entropy and Adam
% SSCB = two Spatial axes (frequency,time), Channels, Batch.
history = table((1:training.MaxEpochs).',nan(training.MaxEpochs,1), ...
    nan(training.MaxEpochs,1),nan(training.MaxEpochs,1), ...
    'VariableNames',{'Epoch','TrainingLoss','ValidationLoss','ValidationBER'});
averageGradient = []; averageSquaredGradient = []; iteration = 0;
bestLoss = Inf;
for epoch = 1:training.MaxEpochs
    if epoch>1
        [Z,bitLabels,bitMask] = makeDataset(baseCfg,trainingPlan(trainingPlan.Epoch==epoch,:));
    end
    order = randperm(size(Z,4));
    totalLoss = 0; totalBits = 0;
    for first = 1:training.MiniBatchSize:numel(order)
        indices = order(first:min(first+training.MiniBatchSize-1,numel(order)));
        dlZ = batchArray(Z(:,:,:,indices),useGPU);
        dlTargets = batchArray(bitLabels(:,:,:,indices),useGPU);
        dlMask = batchArray(bitMask(:,:,:,indices),useGPU);
        [loss,gradients,state] = dlfeval(@modelLoss,net,dlZ,dlTargets,dlMask);
        lossValue = double(gather(extractdata(loss)));
        assert(isfinite(lossValue),'Nonfinite training loss. Check data/learning rate.');
        net.State = state; % Keep batch-normalization running statistics.
        iteration = iteration+1;
        [net,averageGradient,averageSquaredGradient] = adamupdate(net,gradients, ...
            averageGradient,averageSquaredGradient,iteration,training.LearningRate);
        validBits = nnz(bitMask(:,:,:,indices));
        totalLoss = totalLoss+lossValue*validBits;
        totalBits = totalBits+validBits;
    end
    [validationLoss,validationBER] = evaluate(net,validationZ,validationLabels, ...
        validationMask,training.MiniBatchSize,useGPU);
    history.TrainingLoss(epoch) = totalLoss/totalBits;
    history.ValidationLoss(epoch) = validationLoss;
    history.ValidationBER(epoch) = validationBER;
    fprintf('Epoch %d/%d: train loss %.4f, validation loss %.4f, BER %.4g\n', ...
        epoch,training.MaxEpochs,history.TrainingLoss(epoch),validationLoss,validationBER);
    % Select on validation loss, NEVER on the final test seed.
    if validationLoss<bestLoss
        bestLoss = validationLoss;
        model.Net = transferNetwork(net,@gather); % CPU-portable checkpoint.
        model.BaseConfig = baseCfg;
        model.Training = training;
        model.BestEpoch = epoch;
        model.InputSize = [size(Z,1),size(Z,2),size(Z,3)];
        model.Modulation = baseCfg.Modulation;
        model.ScoreConvention = 'positive means bit 1';
        model.FeatureOrder = 'real([Y,Xp,Hraw]), imag([Y,Xp,Hraw])';
        model.TrainingSamples = trainingPlan(trainingPlan.Epoch<=epoch,:);
        model.ValidationSamples = validationSamples;
        save(fullfile(folder,'trained_network.mat'),'model','history','-v7.3');
    end
end
net = model.Net; % Return the best validation checkpoint, not the last epoch.
save(fullfile(folder,'trained_network.mat'),'model','history','-v7.3');
fprintf('Saved trained_network.mat; best validation epoch: %d.\n',model.BestEpoch);

%% 5. Training plots
if training.ShowPlots
    fig = figure('Name','Mini DeepRx training');
    subplot(1,2,1);
    plot(history.Epoch,history.TrainingLoss,'-o', ...
        history.Epoch,history.ValidationLoss,'-s'); grid on;
    xlabel('Epoch'); ylabel('Masked binary cross-entropy');
    legend('Training','Validation','Location','best');
    subplot(1,2,2);
    plot(history.Epoch,history.ValidationBER,'-o'); grid on;
    xlabel('Epoch'); ylabel('Validation uncoded BER');
    exportgraphics(fig,fullfile(folder,'network_training.png'),'Resolution',150);
end
end

%% Local helpers
function samples = makePlan(roots,snrs,dopplers,slotsPerRoot,epochs,excluded)
    % Choose unique generator seeds; validation excludes every training seed.
    count = numel(roots)*slotsPerRoot*epochs;
    settings = zeros(count,5); used = excluded(:); k = 0;
    for root = roots(:).'
        stream = RandStream('mt19937ar','Seed',root);
        for epoch = 1:epochs
            for slot = 1:slotsPerRoot
                seed = randi(stream,[0 2^32-1]);
                while ismember(seed,used), seed = randi(stream,[0 2^32-1]); end
                used(end+1,1) = seed; %#ok<AGROW>
                % Cycle through conditions, but use a NEW seed for each slot.
                condition = mod((epoch-1)*slotsPerRoot+slot-1,numel(snrs)*numel(dopplers))+1;
                [i,j] = ind2sub([numel(snrs),numel(dopplers)],condition);
                k = k+1; settings(k,:) = [epoch,root,seed,snrs(i),dopplers(j)];
            end
        end
    end
    samples = array2table(settings,'VariableNames',{'Epoch','MasterSeed','Seed','SNRdB','DopplerHz'});
end

function [Z,labels,mask] = makeDataset(baseCfg,samples)
    count = height(samples);
    F = 12*baseCfg.NumResourceBlocks; N = 14;
    q = 2*(1+strcmp(baseCfg.Modulation,'16QAM')); % Bits per symbol.
    Z = zeros(F,N,4*baseCfg.NumReceiveAntennas+2,count,'single');
    labels = zeros(F,N,q,count,'single'); mask = false(F,N,q,count);
    for k = 1:count
        cfg = baseCfg; cfg.Seed = samples.Seed(k);
        cfg.SNRdB = samples.SNRdB(k); cfg.MaximumDopplerShift = samples.DopplerHz(k);
        [X,~,Y,meta] = mini_simo_tdl(cfg);
        [Z(:,:,:,k),labels(:,:,:,k),mask(:,:,:,k)] = deeprx_sample(Y,X,meta);
    end
end

function net = buildCNN(inputSize,numBits)
    % DeepRx Table I: 64-wide input conv, eleven residual blocks,
    % depth multiplier 2, and B = numBits output channels (4 for 16QAM).
    % These are (frequency,time) dilations, not the paper's (time,frequency).
    dilations = [1 1; 1 1; 3 2; 3 2; 3 2; 6 3; 3 2; 3 2; 3 2; 1 1; 1 1];
    widths = [64 64 128 128 256 256 256 128 128 64 64];
    graph = layerGraph([
        imageInputLayer(inputSize,'Normalization','none','Name','input')
        convolution2dLayer(3,widths(1),'Padding','same','Name','input_conv')]);
    previous = 'input_conv'; previousWidth = widths(1);
    for b = 1:size(dilations,1)
        prefix = sprintf('block%d_',b);
        block = [
            batchNormalizationLayer('Name',[prefix 'bn1'])
            reluLayer('Name',[prefix 'relu1'])
            groupedConvolution2dLayer(3,2,'channel-wise','Padding','same', ...
                'DilationFactor',dilations(b,:),'Name',[prefix 'depth1'])
            convolution2dLayer(1,widths(b),'Name',[prefix 'point1'])
            batchNormalizationLayer('Name',[prefix 'bn2'])
            reluLayer('Name',[prefix 'relu2'])
            groupedConvolution2dLayer(3,2,'channel-wise','Padding','same', ...
                'DilationFactor',dilations(b,:),'Name',[prefix 'depth2'])
            convolution2dLayer(1,widths(b),'Name',[prefix 'point2'])
            additionLayer(2,'Name',[prefix 'add'])];
        graph = addLayers(graph,block);
        graph = connectLayers(graph,previous,[prefix 'bn1']);
        if widths(b)==previousWidth
            graph = connectLayers(graph,previous,[prefix 'add/in2']);
        else
            % 1x1 projection so the skip path matches the new width.
            graph = addLayers(graph,convolution2dLayer(1,widths(b),'Name',[prefix 'skip']));
            graph = connectLayers(graph,previous,[prefix 'skip']);
            graph = connectLayers(graph,[prefix 'skip'],[prefix 'add/in2']);
        end
        previous = [prefix 'add']; previousWidth = widths(b);
    end
    graph = addLayers(graph,convolution2dLayer(3,numBits,'Padding','same','Name','bit_scores'));
    graph = connectLayers(graph,previous,'bit_scores');
    net = dlnetwork(graph);
end

function array = batchArray(array,useGPU)
    array = single(array);
    if useGPU, array = gpuArray(array); end
    array = dlarray(array,'SSCB');
end

function net = transferNetwork(net,operation)
    net = dlupdate(operation,net);
    state = net.State;
    state.Value = cellfun(operation,state.Value,'UniformOutput',false);
    net.State = state;
end

function [loss,gradients,state] = modelLoss(net,Z,targets,mask)
    [scores,state] = forward(net,Z); % Updates batch-normalization statistics.
    % Stable BCE from logits; sigmoid(scores) is P(bit=1).
    % No sigmoid output layer or softmax over the two independent bits.
    perBitLoss = lossPerBit(scores,targets);
    loss = sum(perBitLoss.*mask,'all')/sum(mask,'all');
    gradients = dlgradient(loss,net.Learnables);
end

function [loss,ber] = evaluate(net,Z,targets,mask,batchSize,useGPU)
    lossSum = 0; errors = 0; numBits = 0;
    for first = 1:batchSize:size(Z,4)
        indices = first:min(first+batchSize-1,size(Z,4));
        scores = gather(extractdata(predict(net,batchArray(Z(:,:,:,indices),useGPU))));
        target = targets(:,:,:,indices); valid = mask(:,:,:,indices);
        perBitLoss = lossPerBit(scores,target);
        lossSum = lossSum+sum(double(perBitLoss(valid)));
        errors = errors+nnz((scores(valid)>0)~=target(valid));
        numBits = numBits+nnz(valid);
    end
    loss = lossSum/numBits; ber = errors/numBits;
    assert(isfinite(loss),'Nonfinite validation loss.');
end

function loss = lossPerBit(scores,targets)
    % Stable log(1+exp(scores)) without the abs() gradient ambiguity at zero.
    % The derivative through the shift cancels, even at scores == 0.
    shift = max(scores,0);
    loss = shift-scores.*targets+log(exp(-shift)+exp(scores-shift));
end

function verifyLossGradient()
    scores = dlarray(single([-100 -3 -1 0 0 1 3 100]));
    targets = dlarray(single([0 0 1 0 1 1 0 1]));
    mask = dlarray(single([1 1 1 1 1 1 0 1]));
    [value,gradient] = dlfeval(@lossProbe,scores,targets,mask);
    expected = (1./(1+exp(-extractdata(scores)))-extractdata(targets)).*extractdata(mask);
    assert(isfinite(extractdata(value)) && ...
        max(abs(extractdata(gradient)-expected))<1e-6, ...
        'Masked BCE gradient check failed.');
    fprintf('Loss-gradient check passed (including zero scores and masked bits).\n');
end

function [value,gradient] = lossProbe(scores,targets,mask)
    value = sum(lossPerBit(scores,targets).*mask,'all');
    gradient = dlgradient(value,scores);
end

