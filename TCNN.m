function [net, history] = TCNN(varargin)
%% Train a time-domain CNN in front of the conventional receiver (TCNN/FEQ)
% Run: TCNN
% Pihlajasalo et al. (2023), Sec. III-C. A residual CNN cleans the received
% TIME samples (before CP removal and FFT). Fixed, untrainable layers then
% do what decode_simo_ls.m does: FFT, LS pilot channel estimate with linear
% interpolation, pilot noise estimate, LMMSE combining, and max-log 16QAM
% demapping. Only the CNN learns, but the loss is the same bit
% cross-entropy as network.m: gradients pass back through the fixed
% receiver, so the CNN learns whatever helps the LS receiver detect bits.
%
% The transmitter PA is ON. Each slot draws one of NumTrainingPAs dithered
% polynomial PAs and an input back-off from PABackoffRange. Validation
% uses its own NumValidationPAs PAs, never seen in training.
% The CNN's last layer starts at zero, so before training the receiver IS
% the LS receiver: epoch 0 in the printout is the LS baseline.
%   TCNN('PABackoffRange',[0 6],'MaxEpochs',50)
%   TCNN('ChannelModels',{'TDL-B','TDL-C','TDL-D'})
%   TCNN('PrepareOnly',true)   % Prepare tcnn_data.mat without training.
% Outputs: tcnn_data.mat, trained_tcnn.mat, tcnn_training.png, tcnn_backoff.png.

%% 1. Editable defaults and optional name/value overrides
p = inputParser; p.PartialMatching = false;
seedList = @(x) isnumeric(x) && isvector(x) && ~isempty(x) && ...
    isreal(x) && all(isfinite(x)) && all(x>=0 & x<=2^32-1 & x==fix(x));
positiveInteger = @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x>=1 && x==fix(x);
addParameter(p,'TrainingSeeds',42:46,seedList);       % FIVE master seeds.
addParameter(p,'ValidationSeeds',1001:1002,seedList); % Never used for gradients.
addParameter(p,'SamplesPerSeed',200,positiveInteger); % Fresh slots per master seed PER EPOCH.
addParameter(p,'ValidationSamplesPerSeed',50,positiveInteger); % Fixed held-out slots.
% Each slot draws SNR, Doppler, SIR, delay spread, and PA back-off
% independently and uniformly from these [min max] ranges.
rangeCheck = @(x) isnumeric(x) && numel(x)==2 && isreal(x) && all(isfinite(x)) && x(1)<=x(2);
addParameter(p,'SNRRange',[-4 32],rangeCheck);
addParameter(p,'DopplerRange',[0 500],@(x) rangeCheck(x) && all(x>=0));
% No interferer by default, so the CNN sees only PA distortion.
addParameter(p,'SIRRange',[Inf Inf],@(x) isequal(x,[Inf Inf]) || rangeCheck(x));
addParameter(p,'ChannelModels',{'TDL-C'},@(x) (iscellstr(x) || isstring(x)) && ~isempty(x));
addParameter(p,'DelaySpreadRange',[10e-9 300e-9],@(x) rangeCheck(x) && all(x>0));
addParameter(p,'PilotConfigs',struct('DMRSAdditionalPosition',0),@(x) isstruct(x) && ~isempty(x));
% PA: input back-off below saturation, as in simo_config. 0 dB gives about
% 27% EVM and 10 dB about 4% with the default polynomial.
addParameter(p,'PABackoffRange',[0 10],rangeCheck);
addParameter(p,'PACoefficients',[1 -1/3],@(x) isnumeric(x) && isvector(x) && all(isfinite(x)));
addParameter(p,'PADither',0.1,@(x) isnumeric(x) && isscalar(x) && isreal(x) && x>=0);
addParameter(p,'NumTrainingPAs',30,positiveInteger);
addParameter(p,'NumValidationPAs',10,positiveInteger);
addParameter(p,'PASeed',2023,@(x) seedList(x) && isscalar(x));
addParameter(p,'MaxEpochs',150,positiveInteger);
addParameter(p,'MiniBatchSize',4,positiveInteger);
addParameter(p,'LearningRate',1e-3,@(x) isnumeric(x) && isscalar(x) && isreal(x) && isfinite(x) && x>0);
addParameter(p,'InitializationSeed',7,@(x) seedList(x) && isscalar(x));
addParameter(p,'ExecutionEnvironment','auto',@(x) any(strcmp(x,{'auto','cpu','gpu'})));
addParameter(p,'PrepareOnly',false,@(x) islogical(x) && isscalar(x));
addParameter(p,'ShowPlots',true,@(x) islogical(x) && isscalar(x));
parse(p,varargin{:}); training = p.Results;
training.ChannelModels = cellstr(training.ChannelModels);
training.PilotConfigs = training.PilotConfigs(:);
training.PACoefficients = training.PACoefficients(:).';
assert(numel(unique(training.TrainingSeeds))==numel(training.TrainingSeeds), ...
    'TrainingSeeds must not contain duplicates.');
assert(numel(unique(training.ValidationSeeds))==numel(training.ValidationSeeds), ...
    'ValidationSeeds must not contain duplicates.');
assert(isempty(intersect(training.TrainingSeeds,training.ValidationSeeds)), ...
    'Training and validation seeds must be different.');
if ~training.PrepareOnly
    assert(~isempty(which('dlnetwork')) && ~isempty(which('groupedConvolution2dLayer')), ...
        ['Training requires Deep Learning Toolbox. ' ...
        'Use TCNN(''PrepareOnly'',true) to prepare the data only.']);
    verifyLossGradient();
end
folder = fileparts(mfilename('fullpath'));
baseCfg = simo_config('SNRdB',20,'MaximumDopplerShift',100, ...
    'PAModel','polynomial','PACoefficients',training.PACoefficients, ...
    'ShowPlots',false,'RunChecks',false,'SaveData',false,'Verbose',false);
for k = 1:numel(training.ChannelModels)
    simo_config('ChannelModel',training.ChannelModels{k});
end

%% 2. PAs, fixed receiver operators, and the slot plan
% Separate PA banks: validation PAs are never used for gradients.
paStream = RandStream('mrg32k3a','Seed',training.PASeed);
paBank.Training = makePABank(training.PACoefficients,training.PADither, ...
    training.NumTrainingPAs,paStream);
paBank.Validation = makePABank(training.PACoefficients,training.PADither, ...
    training.NumValidationPAs,paStream);
[rx,ops] = makeReceiver(baseCfg,training);
roots = [training.TrainingSeeds(:);training.ValidationSeeds(:)];
trainingPlan = makePlan(training.TrainingSeeds,training,training.SamplesPerSeed, ...
    training.MaxEpochs,roots,training.NumTrainingPAs);
validationSamples = makePlan(training.ValidationSeeds,training, ...
    training.ValidationSamplesPerSeed,1,[roots;trainingPlan.Seed],training.NumValidationPAs);
assert(isempty(intersect(trainingPlan.Seed,validationSamples.Seed)));
trainingSamples = trainingPlan(trainingPlan.Epoch==1,:);
[Z,bitLabels,bitMask,pilotIndex] = makeDataset(baseCfg,training,trainingSamples, ...
    paBank.Training,rx,ops);
[validationZ,validationLabels,validationMask,validationPilots] = makeDataset(baseCfg, ...
    training,validationSamples,paBank.Validation,rx,ops);
% Z: time sample x OFDM symbol x [real(Yt) imag(Yt)] x slot (552 x 14 x 4 x 1000),
% scaled by rx.TimeScale to about unit power. Labels/masks as in network.m.
save(fullfile(folder,'tcnn_data.mat'),'Z','bitLabels','bitMask','pilotIndex', ...
    'validationZ','validationLabels','validationMask','validationPilots', ...
    'trainingSamples','trainingPlan','validationSamples','training','baseCfg', ...
    'paBank','rx','ops','-v7.3');
fprintf('Prepared %d training slots and %d validation slots.\n', ...
    size(Z,4),size(validationZ,4));
net = []; history = table();
if training.PrepareOnly
    fprintf('Saved tcnn_data.mat. No network was trained.\n');
    return;
end

%% 3. Build the time-domain CNN (paper Sec. III-C) and check the fixed receiver
verifyReceiver(baseCfg,training,paBank.Validation,rx,ops);
previousRng = rng;
rngCleanup = onCleanup(@() rng(previousRng));
rng(training.InitializationSeed,'twister');
net = buildTCNN([size(Z,1),size(Z,2),size(Z,3)]);
fprintf('TCNN learnable parameters: %d\n', ...
    sum(cellfun(@numel,net.Learnables.Value)));
useGPU = ~strcmp(training.ExecutionEnvironment,'cpu') && canUseGPU;
assert(~strcmp(training.ExecutionEnvironment,'gpu') || useGPU, ...
    'GPU was requested but is unavailable. Select cpu or auto.');
rxDevice = rx;
if useGPU
    net = transferNetwork(net,@gpuArray);
    rxDevice = moveReceiver(rx,@gpuArray);
end
fprintf('Training with GPU: %d. Positive output scores mean bit 1.\n',useGPU);

%% 4. Custom training loop: masked binary cross-entropy and Adam
history = table((0:training.MaxEpochs).',nan(training.MaxEpochs+1,1), ...
    nan(training.MaxEpochs+1,1),nan(training.MaxEpochs+1,1), ...
    'VariableNames',{'Epoch','TrainingLoss','ValidationLoss','ValidationBER'});
% Epoch 0: the untrained CNN outputs its input, so this is the LS receiver.
[validationLoss,validationBER] = evaluate(net,validationZ,validationLabels, ...
    validationMask,validationPilots,training.MiniBatchSize,useGPU,rxDevice,ops);
history.ValidationLoss(1) = validationLoss; history.ValidationBER(1) = validationBER;
fprintf('Epoch 0 (LS receiver): validation loss %.4f, BER %.4g\n',validationLoss,validationBER);
averageGradient = []; averageSquaredGradient = []; iteration = 0;
bestLoss = Inf;
for epoch = 1:training.MaxEpochs
    if epoch>1
        [Z,bitLabels,bitMask,pilotIndex] = makeDataset(baseCfg,training, ...
            trainingPlan(trainingPlan.Epoch==epoch,:),paBank.Training,rx,ops);
    end
    order = randperm(size(Z,4));
    totalLoss = 0; totalBits = 0;
    for first = 1:training.MiniBatchSize:numel(order)
        indices = order(first:min(first+training.MiniBatchSize-1,numel(order)));
        dlZ = batchArray(Z(:,:,:,indices),useGPU);
        targets = deviceArray(bitLabels(:,:,:,indices),useGPU);
        mask = deviceArray(bitMask(:,:,:,indices),useGPU);
        [loss,gradients,state] = dlfeval(@modelLoss,net,dlZ,targets,mask, ...
            rxDevice,ops,pilotIndex(indices));
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
        validationMask,validationPilots,training.MiniBatchSize,useGPU,rxDevice,ops);
    row = epoch+1;
    history.TrainingLoss(row) = totalLoss/totalBits;
    history.ValidationLoss(row) = validationLoss;
    history.ValidationBER(row) = validationBER;
    fprintf('Epoch %d/%d: train loss %.4f, validation loss %.4f, BER %.4g\n', ...
        epoch,training.MaxEpochs,history.TrainingLoss(row),validationLoss,validationBER);
    % Select on validation loss, NEVER on test data.
    if validationLoss<bestLoss
        bestLoss = validationLoss;
        model.Net = transferNetwork(net,@gather); % CPU-portable checkpoint.
        model.Receiver = rx;
        model.PilotOperators = ops;
        model.PABank = paBank;
        model.BaseConfig = baseCfg;
        model.Training = training;
        model.BestEpoch = epoch;
        model.InputSize = [size(Z,1),size(Z,2),size(Z,3)];
        model.Modulation = baseCfg.Modulation;
        model.ScoreConvention = 'positive means bit 1';
        model.FeatureOrder = 'TimeScale*[real(Yt) imag(Yt)], one channel per antenna';
        model.TrainingSamples = trainingPlan(trainingPlan.Epoch<=epoch,:);
        model.ValidationSamples = validationSamples;
        save(fullfile(folder,'trained_tcnn.mat'),'model','history','-v7.3');
    end
end
net = model.Net; % Return the best validation checkpoint, not the last epoch.

%% 5. Validation BER versus PA back-off: LS receiver versus TCNN/FEQ
bestNet = net;
if useGPU, bestNet = transferNetwork(net,@gpuArray); end
[~,~,errorsTCNN,bitsPerSlot] = evaluate(bestNet,validationZ,validationLabels, ...
    validationMask,validationPilots,training.MiniBatchSize,useGPU,rxDevice,ops);
[~,~,errorsLS] = evaluate([],validationZ,validationLabels, ...
    validationMask,validationPilots,training.MiniBatchSize,useGPU,rxDevice,ops);
% One-dB bins of the drawn back-off; pool errors and bits within each bin.
edges = floor(training.PABackoffRange(1)):ceil(training.PABackoffRange(2));
if isscalar(edges), edges = edges+[0 1]; end
bin = discretize(validationSamples.PABackoffdB,edges,'IncludedEdge','left');
bin(isnan(bin)) = numel(edges)-1; % The upper limit joins the last bin.
binBits = accumarray(bin,bitsPerSlot,[numel(edges)-1 1]);
backoffResults = table(edges(1:end-1).'+0.5,accumarray(bin,1,[numel(edges)-1 1]), ...
    binBits,accumarray(bin,errorsLS,[numel(edges)-1 1])./binBits, ...
    accumarray(bin,errorsTCNN,[numel(edges)-1 1])./binBits, ...
    'VariableNames',{'BackoffdB','Slots','TotalBits','BERLS','BERTCNN'});
disp('Validation BER versus PA back-off (unseen PAs):'); disp(backoffResults);
save(fullfile(folder,'trained_tcnn.mat'),'model','history','backoffResults','-v7.3');
fprintf('Saved trained_tcnn.mat; best validation epoch: %d.\n',model.BestEpoch);

%% 6. Training and back-off plots
if training.ShowPlots
    fig = figure('Name','TCNN/FEQ training');
    subplot(1,2,1);
    plot(history.Epoch,history.TrainingLoss,'-o', ...
        history.Epoch,history.ValidationLoss,'-s'); grid on;
    xlabel('Epoch (0 = LS receiver)'); ylabel('Masked binary cross-entropy');
    legend('Training','Validation','Location','best');
    subplot(1,2,2);
    plot(history.Epoch,history.ValidationBER,'-o'); grid on; hold on;
    yline(history.ValidationBER(1),'--','LS receiver');
    xlabel('Epoch'); ylabel('Validation uncoded BER');
    exportgraphics(fig,fullfile(folder,'tcnn_training.png'),'Resolution',150);
    fig = figure('Name','TCNN/FEQ versus PA back-off');
    valid = backoffResults.TotalBits>0;
    semilogy(backoffResults.BackoffdB(valid),backoffResults.BERLS(valid),'-o', ...
        backoffResults.BackoffdB(valid),backoffResults.BERTCNN(valid),'-s', ...
        'LineWidth',1.5); grid on;
    xlabel('PA input back-off (dB), 1 dB bins'); ylabel('Validation uncoded BER');
    title('Unseen PAs, SNR and channel drawn per slot');
    legend('LS receiver','TCNN/FEQ','Location','best');
    exportgraphics(fig,fullfile(folder,'tcnn_backoff.png'),'Resolution',150);
end
end

%% Local helpers: data
function bank = makePABank(base,dither,count,stream)
    % Paper Sec. IV-A: add Gaussian noise to every polynomial coefficient,
    % proportional to its magnitude. Complex noise also adds AM/PM. Keep a
    % draw only if its AM/AM curve rises and bends down (first derivative
    % positive, second negative) up to saturation |u| = 1, like a real PA.
    u = linspace(0,1,201);
    powers = u(:).^(2*(1:numel(base))-1); % Odd orders u, u^3, ...
    bank = zeros(count,numel(base)); k = 0; attempts = 0;
    while k<count
        attempts = attempts+1;
        assert(attempts<=1000*count,'Could not draw valid PAs; reduce PADither.');
        noise = (randn(stream,size(base))+1i*randn(stream,size(base)))/sqrt(2);
        a = base+dither*abs(base).*noise;
        amam = abs(powers*a(:));
        if all(diff(amam)>0) && all(diff(amam,2)<1e-12)
            k = k+1; bank(k,:) = a;
        end
    end
end

function samples = makePlan(roots,training,slotsPerRoot,epochs,excluded,numPAs)
    % As network.m, plus a PA back-off and a PA index for every slot.
    uniform = @(stream,range) range(1)+diff(range)*rand(stream);
    count = numel(roots)*slotsPerRoot*epochs;
    settings = zeros(count,11); used = excluded(:); k = 0;
    for root = roots(:).'
        stream = RandStream('mt19937ar','Seed',root);
        conditionStream = RandStream('mrg32k3a','Seed',root);
        for epoch = 1:epochs
            for slot = 1:slotsPerRoot
                seed = randi(stream,[0 2^32-1]);
                while ismember(seed,used), seed = randi(stream,[0 2^32-1]); end
                used(end+1,1) = seed; %#ok<AGROW>
                model = randi(conditionStream,numel(training.ChannelModels));
                pilot = randi(conditionStream,numel(training.PilotConfigs));
                delaySpread = uniform(conditionStream,training.DelaySpreadRange);
                snr = uniform(conditionStream,training.SNRRange);
                doppler = uniform(conditionStream,training.DopplerRange);
                if isequal(training.SIRRange,[Inf Inf]), sir = Inf;
                else, sir = uniform(conditionStream,training.SIRRange); end
                backoff = uniform(conditionStream,training.PABackoffRange);
                pa = randi(conditionStream,numPAs);
                k = k+1; settings(k,:) = [epoch,root,seed,snr,doppler, ...
                    model,delaySpread,pilot,sir,backoff,pa];
            end
        end
    end
    samples = array2table(settings,'VariableNames',{'Epoch','MasterSeed','Seed', ...
        'SNRdB','DopplerHz','ChannelIndex','DelaySpread','PilotIndex','SIRdB', ...
        'PABackoffdB','PAIndex'});
    models = string(training.ChannelModels(samples.ChannelIndex));
    samples.ChannelModel = models(:);
end

function [Z,labels,mask,pilotIndex] = makeDataset(baseCfg,training,samples,paBank,rx,ops)
    count = height(samples);
    F = 12*baseCfg.NumResourceBlocks; N = 14;
    q = 2*(1+strcmp(baseCfg.Modulation,'16QAM')); % Bits per symbol.
    Z = zeros(rx.NumRows,N,2*baseCfg.NumReceiveAntennas,count,'single');
    labels = zeros(F,N,q,count,'single'); mask = false(F,N,q,count);
    pilotIndex = samples.PilotIndex;
    for k = 1:count
        cfg = slot_config(baseCfg,training.PilotConfigs(samples.PilotIndex(k)));
        cfg.Seed = samples.Seed(k);
        cfg.SNRdB = samples.SNRdB(k); cfg.MaximumDopplerShift = samples.DopplerHz(k);
        cfg.ChannelModel = char(samples.ChannelModel(k));
        cfg.DelaySpread = samples.DelaySpread(k);
        cfg.SIRdB = samples.SIRdB(k);
        cfg.PABackoffdB = samples.PABackoffdB(k);
        cfg.PACoefficients = paBank(samples.PAIndex(k),:);
        [X,~,Y,meta] = mini_simo_channel(cfg);
        op = ops(samples.PilotIndex(k));
        assert(isequal(meta.PilotMask(op.Rows,op.Cols),true(numel(op.Rows),numel(op.Cols))) && ...
            nnz(meta.PilotMask)==numel(op.Pilots) && ...
            max(abs(X(op.Rows,op.Cols)-op.Pilots),[],'all')<1e-12, ...
            'Slot pilots differ from the receiver''s pilot operator.');
        Z(:,:,:,k) = timeFeatures(meta.TimeDomainY,rx);
        [~,labels(:,:,:,k),mask(:,:,:,k)] = deeprx_sample(Y,X,meta);
    end
end

function Z = timeFeatures(Yt,rx)
    % Real and imaginary parts as channels, scaled to about unit power.
    Z = single(rx.TimeScale*cat(3,real(Yt),imag(Yt)));
    assert(all(isfinite(Z(:))),'Network inputs must be finite.');
end

%% Local helpers: the fixed receiver after the CNN
function [rx,ops] = makeReceiver(baseCfg,training)
    % Fixed operators from one noiseless, linear slot per pilot layout.
    % The FFT mapping and the pilots depend only on the configuration.
    for k = 1:numel(training.PilotConfigs)
        cfg = slot_config(baseCfg,training.PilotConfigs(k));
        cfg.PAModel = 'none'; cfg.SNRdB = Inf; cfg.MaximumDopplerShift = 0;
        [X,~,~,meta] = mini_simo_channel(cfg);
        ops(k,1) = pilotOperator(meta.PilotMask,X); %#ok<AGROW>
        g = meta.TimeGrid;
        if k==1
            F = size(X,1); Nfft = g.Nfft;
            rx.NumRows = size(meta.TimeDomainY,1);
            rx.FFTRows = g.FFTRows;
            % Unit-power time samples: the IFFT of F unit-power subcarriers
            % has power F/Nfft^2 per sample.
            rx.TimeScale = Nfft/sqrt(F);
            % FFT of the window, kept to the active bins as a real matrix
            % pair (W = Wr + 1i*Wi), undoing TimeScale. Exact and differentiable.
            W = exp(-2i*pi*(g.ActiveBins-1)*(0:Nfft-1)/Nfft)/rx.TimeScale;
            rx.Wr = single(real(W)); rx.Wi = single(imag(W));
            rx.Pr = single(real(g.PhaseCorrection));
            rx.Pi = single(imag(g.PhaseCorrection));
            rx.NumReceiveAntennas = size(meta.TimeDomainY,3);
            rx.Modulation = baseCfg.Modulation;
            reference = g;
        else
            assert(isequal(g,reference),'Pilot layouts must share the OFDM numerology.');
        end
    end
end

function op = pilotOperator(mask,X)
    % decode_simo_ls.m as matrices: Hls = Frequency * Hpilot * Time.'.
    cols = find(any(mask,1)); rows = find(mask(:,cols(1)));
    assert(isequal(mask(:,cols),repmat(mask(:,cols(1)),1,numel(cols))), ...
        'All pilot symbols must use the same subcarriers.');
    [F,N] = size(mask); P = numel(rows); K = numel(cols);
    op.Rows = rows; op.Cols = cols;
    op.Pilots = X(rows,cols);
    assert(all(abs(abs(op.Pilots(:))-1)<1e-10),'This receiver expects unit-power pilots.');
    % Linear in frequency with linear extrapolation at the band edges.
    op.Frequency = single(interp1(rows,eye(P),(1:F).','linear','extrap'));
    % Linear in time between pilot symbols, held constant outside them.
    if K==1
        op.Time = ones(N,1,'single');
    else
        t = min(max((1:N).',cols(1)),cols(end));
        op.Time = single(interp1(cols,eye(K),t,'linear'));
    end
    % Second differences over equally spaced pilots: E|d|^2 = 6*sigma^2.
    stride = pilotStride(rows); D = zeros(0,P);
    for o = 1:stride
        idx = o:stride:P;
        for j = 1:numel(idx)-2
            D(end+1,idx(j:j+2)) = [1 -2 1]; %#ok<AGROW>
        end
    end
    assert(~isempty(D),'Noise estimation requires at least three equally spaced pilots.');
    op.SecondDifference = single(D);
    op.PilotsR = single(real(op.Pilots)); op.PilotsI = single(imag(op.Pilots));
end

function stride = pilotStride(f)
    % Same rule as decode_simo_ls.m: smallest s with f(o:s:end) equally spaced.
    for stride = 1:numel(f)
        uniform = true;
        for o = 1:stride
            spacing = diff(f(o:stride:end));
            uniform = uniform && all(spacing==spacing(1:min(1,end)));
        end
        if uniform, return; end
    end
end

function scores = feqReceiver(out,rx,ops,pilotIndex)
    % out: time grid x OFDM symbol x [real imag] x slot (unformatted).
    % Returns bit scores (logits, positive = bit 1): subcarrier x symbol x bit x slot.
    Nr = rx.NumReceiveAntennas;
    x = out(rx.FFTRows,:,:,:); % CP removal: the FFT window rows.
    [Nfft,N,~,B] = size(x);
    xr = reshape(x(:,:,1:Nr,:),Nfft,[]); xi = reshape(x(:,:,Nr+1:end,:),Nfft,[]);
    F = size(rx.Wr,1);
    Yr = reshape(rx.Wr*xr-rx.Wi*xi,F,N,Nr,B);
    Yi = reshape(rx.Wr*xi+rx.Wi*xr,F,N,Nr,B);
    [Yr,Yi] = deal(rx.Pr.*Yr-rx.Pi.*Yi,rx.Pr.*Yi+rx.Pi.*Yr); % Undo window shift.
    scores = cell(1,B);
    for b = 1:B
        op = ops(pilotIndex(b));
        yr = Yr(:,:,:,b); yi = Yi(:,:,:,b);
        % Raw LS at pilots: Y.*conj(Xp), pilot x pilot symbol x antenna.
        pr = yr(op.Rows,op.Cols,:); pi_ = yi(op.Rows,op.Cols,:);
        hr = pr.*op.PilotsR+pi_.*op.PilotsI;
        hi = pi_.*op.PilotsR-pr.*op.PilotsI;
        P = numel(op.Rows);
        dr = op.SecondDifference*reshape(hr,P,[]);
        di = op.SecondDifference*reshape(hi,P,[]);
        sigma2 = mean(dr.^2+di.^2,'all')/6;
        % Interpolate per antenna, then unbiased LMMSE (MRC for one stream).
        gain = 0; numR = 0; numI = 0;
        for r = 1:Nr
            Hr = op.Frequency*hr(:,:,r)*op.Time.';
            Hi = op.Frequency*hi(:,:,r)*op.Time.';
            gain = gain+Hr.^2+Hi.^2;
            numR = numR+Hr.*yr(:,:,r)+Hi.*yi(:,:,r);
            numI = numI+Hr.*yi(:,:,r)-Hi.*yr(:,:,r);
        end
        gain = max(gain,eps('single'));
        nu = sigma2./gain; % Noise variance of the equalized symbol.
        scores{b} = demapMaxLog(numR./gain,numI./gain,nu,rx.Modulation);
    end
    scores = cat(4,scores{:});
end

function scores = demapMaxLog(xr,xi,nu,modulation)
    % Max-log LLRs as logits of P(bit = 1), 5G mapping (TS 38.211 5.1):
    % per real dimension, score = (min distance^2 over bit-0 levels
    % - min over bit-1 levels) / nu. Planes: [real sign, imag sign,
    % real magnitude, imag magnitude] as in deeprx_sample.m.
    if strcmp(modulation,'QPSK')
        a = 1/sqrt(2);
        scores = cat(3,4*a*(-xr)./nu,4*a*(-xi)./nu);
        return
    end
    levels = [1 3 -1 -3]/sqrt(10); % Bits (sign,magnitude): 00 01 10 11.
    [sr,mr] = pam4(xr,levels); [si,mi] = pam4(xi,levels);
    scores = cat(3,sr,si,mr,mi)./nu;
end

function [signScore,magnitudeScore] = pam4(x,levels)
    d = cell(1,4);
    for j = 1:4, d{j} = (x-levels(j)).^2; end
    signScore = min(d{1},d{2})-min(d{3},d{4});      % Positive: negative half.
    magnitudeScore = min(d{1},d{3})-min(d{2},d{4}); % Positive: outer level.
end

function rx = moveReceiver(rx,operation)
    for name = {'Wr','Wi','Pr','Pi'}
        rx.(name{1}) = operation(rx.(name{1}));
    end
end

function verifyReceiver(baseCfg,training,paBank,rx,ops)
    % With no CNN, the fixed layers must reproduce decode_simo_ls.m:
    % the same noise estimate and the same hard decisions.
    for k = 1:numel(training.PilotConfigs)
        cfg = slot_config(baseCfg,training.PilotConfigs(k));
        cfg.Seed = 77+k; cfg.SNRdB = 15; cfg.PABackoffdB = 2;
        cfg.PACoefficients = paBank(1,:);
        cfg.ChannelModel = training.ChannelModels{1};
        [data.X,data.H,data.Y,data.meta] = mini_simo_channel(cfg);
        [results,sigma2Hat] = decode_simo_ls(data);
        scores = feqReceiver(dlarray(timeFeatures(data.meta.TimeDomainY,rx)),rx,ops,k);
        [~,targets,valid] = deeprx_sample(data.Y,data.X,data.meta);
        scores = extractdata(scores);
        errors = nnz((scores(valid)>0)~=targets(valid));
        reference = results.BitErrors(strcmp(results.Receiver,'LS H'));
        assert(abs(errors-reference)<=max(2,1e-3*nnz(valid)), ...
            'Fixed receiver gives %d errors; decode_simo_ls gives %d.',errors,reference);
        % Recompute the noise estimate the same way as feqReceiver.
        op = ops(k); Yp = data.Y(op.Rows,op.Cols,:).*conj(op.Pilots);
        d = double(op.SecondDifference)*reshape(Yp,numel(op.Rows),[]);
        assert(abs(mean(abs(d(:)).^2)/6/sigma2Hat-1)<1e-6,'Noise estimate mismatch.');
    end
    fprintf('Fixed receiver matches decode_simo_ls (%d vs %d errors).\n',errors,reference);
end

%% Local helpers: network and training (as in network.m)
function net = buildTCNN(inputSize)
    % Five pre-activation residual blocks, 64-128-256-128-64 wide (paper
    % Sec. III-C), with network.m's depthwise-separable 3x3 convolutions.
    % Axes are (time sample, OFDM symbol). The PA has no memory, so no
    % dilation. A 1x1 layer maps back to [real imag] per antenna and is ADDED
    % to the input: the CNN learns only the correction (post-distortion).
    % That layer starts at zero, so the untrained receiver is exactly LS.
    widths = [64 128 256 128 64];
    graph = layerGraph([
        imageInputLayer(inputSize,'Normalization','none','Name','input')
        convolution2dLayer(3,widths(1),'Padding','same','Name','input_conv')]);
    previous = 'input_conv'; previousWidth = widths(1);
    for b = 1:numel(widths)
        prefix = sprintf('block%d_',b);
        block = [
            batchNormalizationLayer('Name',[prefix 'bn1'])
            reluLayer('Name',[prefix 'relu1'])
            groupedConvolution2dLayer(3,2,'channel-wise','Padding','same','Name',[prefix 'depth1'])
            convolution2dLayer(1,widths(b),'Name',[prefix 'point1'])
            batchNormalizationLayer('Name',[prefix 'bn2'])
            reluLayer('Name',[prefix 'relu2'])
            groupedConvolution2dLayer(3,2,'channel-wise','Padding','same','Name',[prefix 'depth2'])
            convolution2dLayer(1,widths(b),'Name',[prefix 'point2'])
            additionLayer(2,'Name',[prefix 'add'])];
        graph = addLayers(graph,block);
        graph = connectLayers(graph,previous,[prefix 'bn1']);
        if widths(b)==previousWidth
            graph = connectLayers(graph,previous,[prefix 'add/in2']);
        else
            graph = addLayers(graph,convolution2dLayer(1,widths(b),'Name',[prefix 'skip']));
            graph = connectLayers(graph,previous,[prefix 'skip']);
            graph = connectLayers(graph,[prefix 'skip'],[prefix 'add/in2']);
        end
        previous = [prefix 'add']; previousWidth = widths(b);
    end
    graph = addLayers(graph,[
        convolution2dLayer(1,inputSize(3),'WeightsInitializer','zeros', ...
            'BiasInitializer','zeros','Name','correction')
        additionLayer(2,'Name','output')]);
    graph = connectLayers(graph,previous,'correction');
    graph = connectLayers(graph,'input','output/in2');
    net = dlnetwork(graph);
end

function array = batchArray(array,useGPU)
    array = dlarray(deviceArray(array,useGPU),'SSCB');
end

function array = deviceArray(array,useGPU)
    array = single(array);
    if useGPU, array = gpuArray(array); end
end

function net = transferNetwork(net,operation)
    net = dlupdate(operation,net);
    state = net.State;
    state.Value = cellfun(operation,state.Value,'UniformOutput',false);
    net.State = state;
end

function [loss,gradients,state] = modelLoss(net,Z,targets,mask,rx,ops,pilotIndex)
    [out,state] = forward(net,Z); % Updates batch-normalization statistics.
    scores = feqReceiver(stripdims(out),rx,ops,pilotIndex);
    perBitLoss = lossPerBit(scores,targets);
    loss = sum(perBitLoss.*mask,'all')/sum(mask,'all');
    gradients = dlgradient(loss,net.Learnables);
end

function [loss,ber,slotErrors,slotBits] = evaluate(net,Z,targets,mask,pilotIndex, ...
        batchSize,useGPU,rx,ops)
    % net = [] skips the CNN: the plain LS receiver on the same slots.
    lossSum = 0; count = size(Z,4);
    slotErrors = zeros(count,1); slotBits = zeros(count,1);
    for first = 1:batchSize:count
        indices = first:min(first+batchSize-1,count);
        input = batchArray(Z(:,:,:,indices),useGPU);
        if isempty(net), out = input; else, out = predict(net,input); end
        scores = gather(extractdata(feqReceiver(stripdims(out),rx,ops,pilotIndex(indices))));
        target = targets(:,:,:,indices); valid = mask(:,:,:,indices);
        perBitLoss = lossPerBit(scores,target);
        lossSum = lossSum+sum(double(perBitLoss(valid)));
        wrong = ((scores>0)~=target) & valid;
        slotErrors(indices) = squeeze(sum(wrong,[1 2 3]));
        slotBits(indices) = squeeze(sum(valid,[1 2 3]));
    end
    loss = lossSum/sum(slotBits); ber = sum(slotErrors)/sum(slotBits);
    assert(isfinite(loss),'Nonfinite validation loss.');
end

function loss = lossPerBit(scores,targets)
    % Stable log(1+exp(scores)) - scores.*targets, as in network.m.
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
