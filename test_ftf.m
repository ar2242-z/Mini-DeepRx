function passed = test_ftf()
%% Check the frequency-domain hybrid (FTF-CNN) before any long training
% Run: test_ftf        (CPU only, so it can run beside a GPU training; ~6 min)
% 1. The pseudo-time transform: A'*A = I (the FFT back undoes the IFFT
%    exactly), and with Nfft points a transmit grid column becomes that
%    OFDM symbol's waveform without its CP, the samples the PA distorted.
% 2. nn.fcnn is unchanged by moving its blocks to nn.resnet_blocks: the
%    same initial weights as the previous code (1,234,436 parameters), and
%    the FTF-CNN with the default 4/3/4 split has exactly as many.
% 3. Forward pass: F x N x Q logits per slot; gradients of the bit loss
%    reach every parameter array of all three networks, f1 through both
%    transforms.
% 4. A small FTF-CNN learns 8 slots with heavy PA distortion at least as
%    well as an FCNN with the same blocks, slots and steps.
% 5. train_receiver trains, saves and resumes the flow; nn.receive equals
%    the training forward pass.
quiet = {'ShowPlots',false,'RunChecks',false,'SaveData',false,'Verbose',false};
passed = true;
pa = {'Coding','ldpc','PAModel','polynomial','PACoefficients',[1 -1/3]};
root = fileparts(mfilename('fullpath'));

%% 1. Pseudo-time transform
cfg = simo_config(quiet{:},pa{:},'PABackoffdB',3,'SNRdB',20,'Seed',11);
ctx = rx.context(cfg);
A = nn.pseudo_time_basis(ctx,ctx.Nfft);
errI = max(abs(A'*A - eye(ctx.NumSubcarriers)),[],'all');
stream = RandStream('mt19937ar','Seed',cfg.Seed);
[~,txOut] = tx.transmit(cfg,stream);
w = txOut.WaveformBeforePA;
starts = cumsum([0 ctx.CyclicPrefixLengths(1:end-1)+ctx.Nfft]);
errW = 0;
for s = 1:ctx.NumSymbols
    symbol = w(starts(s) + ctx.CyclicPrefixLengths(s) + (1:ctx.Nfft));
    x = A*txOut.Grid(:,s)/sqrt(ctx.Nfft);
    errW = max(errW, max(abs(x - symbol))/max(abs(symbol)));
end
Z = randn(ctx.NumSubcarriers,ctx.NumSymbols,6,2);
back = nn.pair_transform(nn.pair_transform(Z,A),A');
errZ = max(abs(back - Z),[],'all');
passed = report(errI < 1e-12 && errW < 1e-9 && errZ < 1e-12, sprintf( ...
    ['1. pseudo-time: A''A = I (error %.0e); Nfft-point pseudo-time of the Tx grid = Tx waveform ' ...
    'without CP (error %.0e); FFT(IFFT(Z)) = Z on channel pairs (error %.0e)'],errI,errW,errZ)) && passed;

%% 2. nn.fcnn unchanged; FTF-CNN has the FCNN's parameters
inputSize = [ctx.NumSubcarriers ctx.NumSymbols 2*(2*ctx.NumReceiveAntennas + 1)];
rng(7,'twister'); net = nn.fcnn(inputSize,ctx.BitsPerSymbol);
rng(7,'twister'); old = previousFcnn(inputSize,ctx.BitsPerSymbol);
same = isequal(net.Learnables.Layer,old.Learnables.Layer) && ...
    isequal(cellfun(@extractdata,net.Learnables.Value,'UniformOutput',false), ...
    cellfun(@extractdata,old.Learnables.Value,'UniformOutput',false));
nFCNN = countParameters(net);
rng(7,'twister'); nets = nn.ftf_cnn(inputSize,ctx.BitsPerSymbol,ctx.Nfft);
counts = structfun(@countParameters,nets).';
passed = report(same && nFCNN == 1234436 && sum(counts) == nFCNN, sprintf( ...
    '2. nn.fcnn as before (%d parameters, same initial weights); FTF-CNN f1/t/f2 = %d + %d + %d = %d', ...
    nFCNN,counts,sum(counts))) && passed;

%% 3. Forward shape and gradients
[rxWaveform,truth] = run_link(cfg);
[planes,mask] = nn.labels(truth.CodedBits,ctx);
batch.Slots = dlarray(single(rx.sync(rxWaveform,ctx)));
batch.PilotGrid = ctx.PilotGrid; batch.Labels = dlarray(planes); batch.Mask = dlarray(single(mask));
logits = nn.flow_ftf(nets,batch,false,ctx);
okShape = isequal(size(logits,1:3),[ctx.NumSubcarriers ctx.NumSymbols ctx.BitsPerSymbol]);
g = dlfeval(@ftfGradient,nets,batch,ctx);
reached = structfun(@(t) all(cellfun(@(v) all(isfinite(extractdata(v(:)))) && ...
    any(extractdata(v(:))),t.Value)),g);
passed = report(okShape && all(reached), sprintf( ...
    '3. logits %s; gradient finite and non-zero for every parameter array of f1, t and f2', ...
    mat2str(size(logits,1:3)))) && passed;

%% 4. Overfit 8 slots with heavy PA distortion
fprintf('4. Overfit 8 fixed slots: small FTF-CNN, PA back-off 0 dB, 25 dB, TDL-C, 100 Hz\n');
widths = [16 16 32 32 32 16 16]; dilations = [1 1; 3 2; 3 2; 1 1; 1 1; 3 2; 1 1];
ncfg = nn_config('Link',pa,'SNRRange',[25 25],'DopplerRange',[100 100], ...
    'DelaySpreadRange',[300e-9 300e-9],'SIRRange',[Inf Inf],'PABackoffRange',[0 0], ...
    'MaxEpochs',250,'MiniBatchSize',4,'LearningRate',3e-3,'Widths',widths,'Dilations',dilations, ...
    'FTFSplit',[3 2 2],'ExecutionEnvironment','cpu','Verbose',false);
baseCfg = simo_config(quiet{:},ncfg.Link{:});
bank = nn.pa_bank(baseCfg,ncfg);
plan = nn.plan_slots(ncfg,77,8,1,[],bank.Training);
data = nn.make_data(baseCfg,ncfg,plan,true);
rng(ncfg.InitializationSeed,'twister');
params = nn.ftf_cnn(inputSize,ctx.BitsPerSymbol,ctx.Nfft,widths,dilations,ncfg.FTFSplit);
forward = @(p,b,t) nn.flow_ftf(p,b,t,ctx);
clock = tic;
[best,history] = nn.train(params,forward,@(e) data,data,ncfg,fieldnames(params).');
fprintf('        FTF-CNN: epoch 1 loss %.4f, BER %.4f | best loss %.4f, BER %.4f (epoch %d, %.0f s)\n', ...
    history.ValidationLoss(1),history.ValidationBER(1),history.ValidationLoss(best.Epoch), ...
    history.ValidationBER(best.Epoch),best.Epoch,toc(clock));
% Reference: an FCNN with the same blocks, slots and steps. Small networks
% learn slowly at first (BER near 0.5), so compare with it, not a fixed target.
rng(ncfg.InitializationSeed,'twister');
reference = struct('fcnn',nn.fcnn(inputSize,ctx.BitsPerSymbol,widths,dilations));
[bestF,historyF] = nn.train(reference,@(p,b,t) nn.fcnn_forward(p,b,t,ctx),@(e) data,data,ncfg,{'fcnn'});
fprintf('        FCNN:    epoch 1 loss %.4f, BER %.4f | best loss %.4f, BER %.4f (epoch %d)\n', ...
    historyF.ValidationLoss(1),historyF.ValidationBER(1),historyF.ValidationLoss(bestF.Epoch), ...
    historyF.ValidationBER(bestF.Epoch),bestF.Epoch);
ok = history.ValidationLoss(best.Epoch) < 0.7*history.ValidationLoss(1) && ...
    history.ValidationBER(best.Epoch) < 0.5*history.ValidationBER(1) && ...
    history.ValidationLoss(best.Epoch) <= 1.05*historyF.ValidationLoss(bestF.Epoch);
passed = report(ok,['loss below 70% and BER below half of epoch 1, and loss within 5% of ' ...
    'the same-size FCNN''s or lower']) && passed;

%% 5. train_receiver, Resume and nn.receive
files = fullfile(root,'models',{'test_ftf.mat','test_ftf_last.mat'});
cleanup = onCleanup(@() deleteIfExists(files));
small = {'Link',pa,'Flow','ftf','Name','test_ftf','SlotsPerSeed',1,'ValidationSlotsPerSeed',1, ...
    'SIRRange',[Inf Inf],'Widths',[8 8 8 8],'Dilations',ones(4,2),'FTFSplit',[1 2 1], ...
    'MiniBatchSize',2,'ExecutionEnvironment','cpu'};
evalc('[m1,~] = train_receiver(small{:},''MaxEpochs'',1);');
evalc('[m2,h2] = train_receiver(small{:},''MaxEpochs'',2,''Resume'',true);');
llr = nn.receive(rxWaveform,ctx,m2);
b1.Slots = batch.Slots; b1.PilotGrid = ctx.PilotGrid;
llrF = nn.to_llr(double(extractdata(nn.flow_ftf(m2.Nets,b1,false,ctx))),ctx);
err = max(abs(llr - llrF))/max(abs(llrF));
ok = strcmp(m1.Flow,'ftf') && isequal(m2.Trainable,{'ftf_f1','ftf_t','ftf_f2'}) && height(h2) == 2 && ...
    m2.Nets.ftf_t.Layers(1).InputSize(1) == ctx.Nfft && err < 1e-5 && all(isfinite(llr));
passed = report(ok,sprintf(['5. train_receiver saves the ftf flow (three networks, pseudo-time %d points), ' ...
    'resumes to epoch 2; nn.receive = forward pass (error %.0e)'],ctx.Nfft,err)) && passed;
end

%% Local helpers
function g = ftfGradient(params,batch,ctx)
    logits = nn.flow_ftf(params,batch,true,ctx);
    loss = nn.bce_loss(logits,batch.Labels,batch.Mask);
    g = dlgradient(loss,structfun(@(n) n.Learnables,params,'UniformOutput',false));
end

function n = countParameters(net)
    n = sum(cellfun(@numel,net.Learnables.Value));
end

function net = previousFcnn(inputSize,numBits)
    % nn.fcnn before its blocks moved to nn.resnet_blocks (commit 98e14ba).
    [widths,dilations] = nn.fcnn_layout();
    net = dlnetwork;
    net = addLayers(net,[
        imageInputLayer(inputSize,'Normalization','none','Name','input')
        convolution2dLayer(3,widths(1),'Padding','same','Name','input_conv')]);
    previous = 'input_conv'; previousWidth = widths(1);
    for b = 1:numel(widths)
        p = sprintf('block%d_',b);
        net = addLayers(net,[
            batchNormalizationLayer('Name',[p 'bn1'])
            reluLayer('Name',[p 'relu1'])
            groupedConvolution2dLayer(3,2,'channel-wise','Padding','same', ...
                'DilationFactor',dilations(b,:),'Name',[p 'depth1'])
            convolution2dLayer(1,widths(b),'Name',[p 'point1'])
            batchNormalizationLayer('Name',[p 'bn2'])
            reluLayer('Name',[p 'relu2'])
            groupedConvolution2dLayer(3,2,'channel-wise','Padding','same', ...
                'DilationFactor',dilations(b,:),'Name',[p 'depth2'])
            convolution2dLayer(1,widths(b),'Name',[p 'point2'])
            additionLayer(2,'Name',[p 'add'])]);
        net = connectLayers(net,previous,[p 'bn1']);
        if widths(b) == previousWidth
            net = connectLayers(net,previous,[p 'add/in2']);
        else
            net = addLayers(net,convolution2dLayer(1,widths(b),'Name',[p 'skip']));
            net = connectLayers(net,previous,[p 'skip']);
            net = connectLayers(net,[p 'skip'],[p 'add/in2']);
        end
        previous = [p 'add']; previousWidth = widths(b);
    end
    net = addLayers(net,convolution2dLayer(3,numBits,'Padding','same','Name','bit_logits'));
    net = connectLayers(net,previous,'bit_logits');
    net = initialize(net);
end

function deleteIfExists(files)
    for k = 1:numel(files)
        if isfile(files{k}), delete(files{k}); end
    end
end

function ok = report(ok,label)
    if ok, fprintf('  PASS  %s\n',label); else, fprintf('  FAIL  %s\n',label); end
end
