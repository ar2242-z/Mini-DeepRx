function passed = test_tcnn()
%% Check the TCNN flows before any long training
% Run: test_tcnn        (CPU only, so it can run beside a GPU training)
% 1. Starts as the conventional receiver: the untrained TCNN passes the samples through, so the
%    tcnn_feq flow gives the conventional receiver's LLRs.
% 2. Gradients of the bit loss reach every TCNN weight through the fixed
%    conventional receiver (FFT, LS, interpolation, LMMSE equalizer, demapper).
% 3. A small TCNN in front of the conventional receiver learns to undo
%    heavy PA distortion (0 dB back-off) on 8 fixed slots: loss and BER
%    fall below the conventional receiver's.
% 4. nn.receive with the saved flow gives the training forward pass.
% 5. The hybrid starts as the FCNN: an untrained TCNN in front of a trained
%    FCNN gives that FCNN's LLRs. train_receiver runs with InitFrom and
%    Freeze, and leaves the frozen FCNN unchanged.
% 6. The paper TCNN (TCNNArchitecture 'paper'): the per-symbol layout is
%    exact (an identity network returns the slot), gradients reach every
%    weight, and train_receiver / nn.receive run it.

quiet = {'ShowPlots',false,'RunChecks',false,'SaveData',false,'Verbose',false};
passed = true;
pa = {'Coding','ldpc','PAModel','polynomial','PACoefficients',[1 -1/3]};

%% 1. Untrained tcnn_feq equals the conventional receiver
cfg = simo_config(quiet{:},pa{:},'PABackoffdB',3,'SNRdB',20,'Seed',11);
ctx = rx.context(cfg);
rng(7,'twister');
model = struct('Flow','tcnn_feq','TCNNInputSize',[ctx.SlotLength 1 2*ctx.NumReceiveAntennas]);
model.Nets.tcnn = nn.tcnn(model.TCNNInputSize);
[rxWaveform,truth] = run_link(cfg);
llrT = nn.receive(rxWaveform,ctx,model);
llrLS = rx.receive(rxWaveform,ctx,'conventional');
err = max(abs(llrT-llrLS))/max(abs(llrLS));
rT = evaluate(llrT,truth,cfg); rLS = evaluate(llrLS,truth,cfg);
passed = report(err < 1e-4 && rT.UncodedErrors == rLS.UncodedErrors, sprintf( ...
    '1. untrained TCNN + conventional = conventional (LLR error %.1e of max; %d = %d bit errors)', ...
    err,rT.UncodedErrors,rLS.UncodedErrors)) && passed;

%% 2. Gradients reach every TCNN weight
net = model.Nets.tcnn;
% With the zero-initialized correction only it gets a gradient at first;
% give it small weights, as after the first update, to reach the rest.
k = strcmp(net.Learnables.Layer,'correction') & strcmp(net.Learnables.Parameter,'Weights');
net.Learnables.Value{k} = dlarray(0.01*randn(size(net.Learnables.Value{k}),'single'));
[planes,mask] = nn.labels(truth.CodedBits,ctx);
batch.Slots = dlarray(single(rx.sync(rxWaveform,ctx)));
batch.PilotGrid = ctx.PilotGrid; batch.Labels = dlarray(planes); batch.Mask = dlarray(single(mask));
g = dlfeval(@tcnnGradient,struct('tcnn',net),batch,ctx);
reached = cellfun(@(v) all(isfinite(extractdata(v(:)))) && any(extractdata(v(:))), g.Value);
g0 = dlfeval(@tcnnGradient,model.Nets,batch,ctx);
first = cellfun(@(v) any(extractdata(v(:))), g0.Value);
passed = report(all(reached) && any(first(k)), sprintf( ...
    '2. gradient finite and non-zero for all %d TCNN parameter arrays (at start: the correction layer)', ...
    numel(reached))) && passed;

%% 3. Overfit 8 slots with heavy PA distortion
fprintf('3. Overfit 8 fixed slots: small TCNN + conventional, PA back-off 0 dB, 25 dB, TDL-C, 100 Hz\n');
ncfg = nn_config('Link',pa,'SNRRange',[25 25],'DopplerRange',[100 100], ...
    'DelaySpreadRange',[300e-9 300e-9],'SIRRange',[Inf Inf],'PABackoffRange',[0 0], ...
    'MaxEpochs',150,'MiniBatchSize',4,'LearningRate',3e-3,'TCNNWidths',[32 32 32], ...
    'TCNNDilations',[1 2 1],'ExecutionEnvironment','cpu','Verbose',false);
baseCfg = simo_config(quiet{:},ncfg.Link{:});
bank = nn.pa_bank(baseCfg,ncfg);
plan = nn.plan_slots(ncfg,77,8,1,[],bank.Training);
data = nn.make_data(baseCfg,ncfg,plan,true);
rng(ncfg.InitializationSeed,'twister');
params = struct('tcnn',nn.tcnn([ctx.SlotLength 1 2*ctx.NumReceiveAntennas],ncfg.TCNNWidths,ncfg.TCNNDilations));
forward = @(p,b,t) nn.flow_tcnn_feq(p,b,t,ctx);
clock = tic;
[best,history] = nn.train(params,forward,@(e) data,data,ncfg,{'tcnn'});
% Epoch-0 reference: the untrained TCNN is the conventional receiver.
[ls.Loss,ls.BER] = scoreFlow(params,forward,data);
fprintf('        conventional: loss %.4f, BER %.4f | trained: loss %.4f, BER %.4f, BLER %.2f (best epoch %d, %.0f s)\n', ...
    ls.Loss,ls.BER,history.ValidationLoss(best.Epoch),history.ValidationBER(best.Epoch), ...
    history.ValidationBLER(best.Epoch),best.Epoch,toc(clock));
ok = history.ValidationLoss(best.Epoch) < 0.8*ls.Loss && history.ValidationBER(best.Epoch) < 0.8*ls.BER;
passed = report(ok,'loss and BER at least 20% below the conventional receiver on these slots') && passed;

%% 4. nn.receive equals the training forward pass
model = struct('Flow','tcnn_feq','Nets',best.Params,'TCNNInputSize',[ctx.SlotLength 1 2*ctx.NumReceiveAntennas]);
c = nn.slot_cfg(baseCfg,ncfg,plan(1,:));
[w,t] = run_link(c);
llrR = nn.receive(w,ctx,model);
b1.Slots = dlarray(data.Slots(:,:,1)); b1.PilotGrid = data.PilotGrid(:,:,:,1);
llrF = nn.to_llr(extractdata(forward(best.Params,b1,false)),ctx);
err = max(abs(llrR-double(llrF)))/max(abs(llrF));
rR = evaluate(llrR,t,c);
passed = report(err < 1e-4,sprintf('4. nn.receive = training forward pass (error %.1e); this slot BER %.4f', ...
    err,rR.UncodedErrors/rR.UncodedBits)) && passed;

%% 5. Hybrid starts as the FCNN; InitFrom and Freeze
root = fileparts(mfilename('fullpath'));
fcnnFile = fullfile(root,'reference','legacy_fcnn.mat');
legacy = load(fcnnFile,'model');
fcnnModel = struct('Net',legacy.model.Net,'InputSize',legacy.model.InputSize,'FeatureLayout','deeprx-v1');
cfg = simo_config(quiet{:},'Coding','ldpc','SNRdB',20,'Seed',5001);
ctx = rx.context(cfg);
[rxWaveform,truth] = run_link(cfg);
rng(7,'twister');
hybrid = struct('Flow','tcnn_fcnn','TCNNInputSize',[ctx.SlotLength 1 2*ctx.NumReceiveAntennas]);
hybrid.Nets = struct('tcnn',nn.tcnn(hybrid.TCNNInputSize),'fcnn',fcnnModel.Net);
llrH = nn.receive(rxWaveform,ctx,hybrid);
llrF = nn.receive(rxWaveform,ctx,fcnnModel);
err = max(abs(llrH-llrF))/max(abs(llrF));
rH = evaluate(llrH,truth,cfg); rF = evaluate(llrF,truth,cfg);
passed = report(err < 1e-3 && abs(rH.UncodedErrors-rF.UncodedErrors) <= 2, sprintf( ...
    '5. untrained TCNN + FCNN = FCNN (LLR error %.1e of max; %d vs %d bit errors)', ...
    err,rH.UncodedErrors,rF.UncodedErrors)) && passed;
% A one-epoch hybrid run from a saved FCNN with the FCNN frozen.
saved = fullfile(root,'models','test_tcnn_init.mat');
model = fcnnModel; save(saved,'model');
cleanup = onCleanup(@() deleteIfExists({saved,fullfile(root,'models','test_tcnn_hybrid.mat')}));
evalc(['[m,~] = train_receiver(''Link'',pa,''Flow'',''tcnn_fcnn'',''Name'',''test_tcnn_hybrid'',' ...
    '''InitFrom'',saved,''Freeze'',{''fcnn''},''MaxEpochs'',1,''SlotsPerSeed'',1,' ...
    '''ValidationSlotsPerSeed'',1,''TCNNWidths'',[8 8],''TCNNDilations'',[1 1],' ...
    '''ExecutionEnvironment'',''cpu'');']);
same = isequal(extractLearnables(m.Nets.fcnn),extractLearnables(fcnnModel.Net)) && ...
    isequal(extractState(m.Nets.fcnn),extractState(fcnnModel.Net));
L = m.Nets.tcnn.Learnables;
moved = any(extractdata(L.Value{strcmp(L.Layer,'correction') & strcmp(L.Parameter,'Weights')}),'all');
passed = report(same && moved && strcmp(m.Flow,'tcnn_fcnn') && isequal(m.Trainable,{'tcnn'}), ...
    '   train_receiver: FCNN loaded and frozen (weights and statistics unchanged), TCNN trained') && passed;

%% 6. Paper TCNN (TWC 2023, Sec. III-C): per-symbol layout, gradients, training
cfg = simo_config(quiet{:},pa{:},'PABackoffdB',3,'SNRdB',20,'Seed',11);
ctx = rx.context(cfg);
[rxWaveform,truth] = run_link(cfg);
inputSize = nn.tcnn_input_size(ctx,'paper');
% An identity network (1x1 conv, weights eye(2)) must give the slot back:
% the per-symbol packing and unpacking are exact inverses.
identity = dlnetwork([imageInputLayer(inputSize,'Normalization','none','Name','input')
    convolution2dLayer(1,2,'Weights',reshape(eye(2),1,1,2,2),'Bias',zeros(1,1,2),'Name','output')]);
slot = rx.sync(rxWaveform,ctx);
back = extractdata(nn.tcnn_apply(identity,dlarray(single(slot)),false,ctx));
err = max(abs(back(:)-slot(:)))/max(abs(slot(:)));
passed = report(err < 1e-5,sprintf('6. paper layout %s per antenna: identity network returns the slot (error %.1e)', ...
    mat2str(inputSize),err)) && passed;
rng(7,'twister');
net = nn.tcnn_paper(inputSize);
nParams = sum(cellfun(@numel,net.Learnables.Value));
[planes,mask] = nn.labels(truth.CodedBits,ctx);
batch.Slots = dlarray(single(slot));
batch.PilotGrid = ctx.PilotGrid; batch.Labels = dlarray(planes); batch.Mask = dlarray(single(mask));
g = dlfeval(@tcnnGradient,struct('tcnn',net),batch,ctx);
reached = cellfun(@(v) all(isfinite(extractdata(v(:)))) && any(extractdata(v(:))), g.Value);
passed = report(all(reached),sprintf( ...
    '   gradient finite and non-zero for all %d parameter arrays (%d parameters, default 64-128-256-128-64)', ...
    numel(reached),nParams)) && passed;
evalc(['[m,~] = train_receiver(''Link'',pa,''Flow'',''tcnn_feq'',''TCNNArchitecture'',''paper'',' ...
    '''TCNNWidths'',[8 16],''Name'',''test_tcnn_paper'',''MaxEpochs'',1,''SlotsPerSeed'',1,' ...
    '''ValidationSlotsPerSeed'',1,''SIRRange'',[Inf Inf],''ExecutionEnvironment'',''cpu'');']);
cleanupPaper = onCleanup(@() deleteIfExists({fullfile(root,'models','test_tcnn_paper.mat')}));
llrP = nn.receive(rxWaveform,ctx,m);
passed = report(strcmp(m.TCNNArchitecture,'paper') && isequal(m.TCNNInputSize,inputSize) && ...
    numel(llrP) == numel(llrLS) && all(isfinite(llrP)), ...
    '   train_receiver with TCNNArchitecture paper trains, saves, and nn.receive runs it') && passed;

if passed, fprintf('All TCNN checks passed.\n'); else, fprintf('Some TCNN checks FAILED.\n'); end
end

%% Local helpers
function g = tcnnGradient(params,batch,ctx)
    logits = nn.flow_tcnn_feq(params,batch,{'tcnn'},ctx);
    loss = nn.bce_loss(logits,batch.Labels,batch.Mask);
    g = dlgradient(loss,params.tcnn.Learnables);
end

function [loss,ber] = scoreFlow(params,forward,data)
    lossSum = 0; errors = 0; bits = 0;
    for s = 1:size(data.Slots,3)
        b.Slots = dlarray(data.Slots(:,:,s)); b.PilotGrid = data.PilotGrid(:,:,:,s);
        logits = forward(params,b,false);
        m = data.Mask(:,:,:,s); l = data.Labels(:,:,:,s);
        lossSum = lossSum + double(extractdata(nn.bce_loss(logits,dlarray(l),dlarray(single(m)))))*nnz(m);
        x = extractdata(logits);
        errors = errors + nnz((x(m) > 0) ~= l(m)); bits = bits + nnz(m);
    end
    loss = lossSum/bits; ber = errors/bits;
end

function v = extractLearnables(net)
    v = cellfun(@extractdata,net.Learnables.Value,'UniformOutput',false);
end

function v = extractState(net)
    v = cellfun(@(x) double(gather(x)),net.State.Value,'UniformOutput',false);
end

function deleteIfExists(files)
    for k = 1:numel(files)
        if isfile(files{k}), delete(files{k}); end
    end
end

function ok = report(ok,label)
    if ok, fprintf('  PASS  %s\n',label); else, fprintf('  FAIL  %s\n',label); end
end
