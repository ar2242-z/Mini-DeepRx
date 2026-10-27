function passed = test_nn(referenceFile)
%% Check the neural receiver pipeline (+nn) before any long training
% Run: test_nn
% 1. Oracle round trip: nn.labels -> perfect logits -> nn.to_llr -> evaluate
%    gives zero bit errors and a passing CRC (bit order and LLR sign).
% 2. nn.deeprx_features and nn.labels equal the earlier deeprx_sample.m
%    on the six saved reference slots.
% 3. nn.bce_loss gradient equals sigmoid(s) - t, also at s = 0 and masked.
% 4. nn.fcnn builds the paper's network (same parameter count as the
%    earlier one), and gradients reach the received samples through the
%    features and the FFT (needed for a TCNN in front of it).
% 5. nn.train on 8 fixed slots with a small network: loss and BER fall
%    steadily, so the whole training pipeline can learn (about 40 s on GPU).
% 6. The earlier trained network (reference/legacy_fcnn.mat) runs in the new pipeline (nn.receive)
%    and beats the conventional receiver on new slots (TDL-C, 20 dB, 100 Hz).

if nargin == 0
    referenceFile = fullfile(fileparts(mfilename('fullpath')),'reference','ref_v0.mat');
end
quiet = {'ShowPlots',false,'RunChecks',false,'SaveData',false,'Verbose',false};
passed = true;

%% 1. Oracle round trip
fprintf('1. Labels -> oracle logits -> LLRs -> LDPC decoding\n');
cases = {{'Coding','ldpc'}, {'Coding','ldpc','Modulation','QPSK','DMRSAdditionalPosition',1}};
for k = 1:numel(cases)
    cfg = simo_config(quiet{:},cases{k}{:},'SNRdB',Inf);
    ctx = rx.context(cfg);
    [~,truth] = run_link(cfg);
    [planes,mask] = nn.labels(truth.CodedBits,ctx);
    logits = 10*(2*planes-1);              % Confident and correct.
    llr = nn.to_llr(logits,ctx);
    r = evaluate(llr,truth,cfg);
    ok = r.UncodedErrors==0 && r.CodedErrors==0 && r.BlockError==0 && ...
        all(sign(llr) == 1-2*truth.CodedBits) && nnz(mask)==numel(truth.CodedBits);
    passed = report(ok,sprintf('%-5s %d DMRS symbol(s): 0 errors, CRC passes, LLR > 0 means bit 0', ...
        cfg.Modulation,numel(ctx.PilotCols))) && passed;
end
cfg2 = cfg; cfg2.Seed = 3;
[~,truth2] = run_link(cfg2);
planes2 = nn.labels([truth.CodedBits truth2.CodedBits],ctx);
llr2 = nn.to_llr(10*(2*planes2-1),ctx);
passed = report(isequal(sign(llr2),1-2*[truth.CodedBits truth2.CodedBits]), ...
    'batch of 2 slots maps back in order') && passed;

%% 2. Features and labels against the earlier deeprx_sample
fprintf('2. Features and labels versus legacy deeprx_sample (%s)\n',referenceFile);
saved = load(referenceFile,'ref');
for c = fieldnames(saved.ref.slots).'
    s = saved.ref.slots.(c{1});
    options = [fieldnames(s.cfg).'; struct2cell(s.cfg).'];
    ctx = rx.context(simo_config(options{:}));
    Z = nn.deeprx_features(s.rx.Y,ctx.PilotGrid);
    [planes,mask] = nn.labels(s.tx.Bits,ctx);
    ok = isequal(Z,s.deeprx.Z) && isequal(planes,s.deeprx.Labels) && isequal(mask,s.deeprx.Mask);
    passed = report(ok,sprintf('%-15s Z %s, labels and mask identical',c{1},mat2str(size(Z)))) && passed;
end

%% 3. Loss gradient
s = dlarray(single([-100 -3 -1 0 0 1 3 100]));
t = dlarray(single([0 0 1 0 1 1 0 1]));
m = dlarray(single([1 1 1 1 1 1 0 1]));
[value,g] = dlfeval(@lossProbe,s,t,m);
expected = (1./(1+exp(-extractdata(s)))-extractdata(t)).*extractdata(m)/sum(extractdata(m));
err = max(abs(extractdata(g)-expected));
passed = report(isfinite(extractdata(value)) && err<1e-6, ...
    sprintf('3. BCE gradient equals (sigmoid(s)-t)*mask/N (max error %.1e)',err)) && passed;

%% 4. The paper's network, and gradients back to the received samples
cfg = simo_config(quiet{:},'Coding','ldpc','SNRdB',15);
ctx = rx.context(cfg);
inputSize = [ctx.NumSubcarriers ctx.NumSymbols 2*(2*cfg.NumReceiveAntennas+1)];
rng(7,'twister');
net = nn.fcnn(inputSize,ctx.BitsPerSymbol);
numParams = sum(cellfun(@numel,net.Learnables.Value));
legacy = load(fullfile(fileparts(mfilename('fullpath')),'reference','legacy_fcnn.mat'),'model');
legacyParams = sum(cellfun(@numel,legacy.model.Net.Learnables.Value));
[rxWaveform,truth] = run_link(cfg);
[planes,mask] = nn.labels(truth.CodedBits,ctx);
batch.Slots = dlarray(single(rx.sync(rxWaveform,ctx)));
batch.PilotGrid = ctx.PilotGrid;
batch.Labels = planes; batch.Mask = single(mask);
params.fcnn = net;
[logits,gSlots] = dlfeval(@inputGradient,params,batch,ctx);
g = extractdata(gSlots);
ok = numParams==legacyParams && isequal(size(logits),[inputSize(1:2) ctx.BitsPerSymbol]) && ...
    all(isfinite(g(:))) && any(g(:));
passed = report(ok,sprintf(['4. Table I network: %d parameters (earlier network %d); ' ...
    'output %s; gradient reaches the slot samples'],numParams,legacyParams,mat2str(size(logits)))) && passed;

%% 5. Overfit 8 slots with a small network
fprintf('5. Overfit 8 fixed slots (small network, 20 dB, TDL-C, 100 Hz)\n');
ncfg = nn_config('Link',{'Coding','ldpc'},'SNRRange',[20 20],'DopplerRange',[100 100], ...
    'DelaySpreadRange',[300e-9 300e-9],'SIRRange',[Inf Inf],'MaxEpochs',200, ...
    'MiniBatchSize',4,'LearningRate',3e-3,'Widths',[32 32 32 32 32], ...
    'Dilations',[1 1;3 3;6 6;3 3;1 1],'Verbose',false);
% The dilations give a reach of about 21 OFDM symbols, so every data symbol
% can see the single pilot symbol (column 3); fewer blocks cannot.
baseCfg = simo_config(quiet{:},ncfg.Link{:});
plan = nn.plan_slots(ncfg,77,8,1,[]);
data = nn.make_data(baseCfg,ncfg,plan,true);
ls = 0;
for k = 1:8
    c = nn.slot_cfg(baseCfg,ncfg,plan(k,:));
    [w,tr] = run_link(c);
    r = evaluate(rx.receive(w,rx.context(c),'conventional'),tr,c);
    ls = ls + r.UncodedErrors/r.UncodedBits/8;
end
rng(ncfg.InitializationSeed,'twister');
params = struct('fcnn',nn.fcnn(inputSize,ctx.BitsPerSymbol,ncfg.Widths,ncfg.Dilations));
[best,history] = nn.train(params,@(p,b,t) nn.fcnn_forward(p,b,t,ctx),@(e) data,data,ncfg);
first = history.TrainingLoss(1); last = history.TrainingLoss(end);
fprintf('        training loss %.4f -> %.4f; BER on these slots %.4f (conventional %.4f); best epoch %d\n', ...
    first,last,history.ValidationBER(best.Epoch),ls,best.Epoch);
% 400 Adam steps: enough to prove learning, not to beat the conventional receiver (that takes a
% full training run). BER must fall well below guessing and keep falling.
ber = history.ValidationBER;
ok = last < 0.4*first && ber(end) < 0.25 && ber(end) < ber(50);
passed = report(ok,'loss below 40% of its start; BER below 0.25 and still falling') && passed;

%% 6. Earlier trained network in the new pipeline
fprintf('6. Earlier trained network (reference/legacy_fcnn.mat) through nn.receive (TDL-C, 20 dB, 100 Hz, 20 slots)\n');
model = legacyModel(legacy.model);
counts = zeros(2,3);
for seed = 5001:5020
    cfg = simo_config(quiet{:},'Coding','ldpc','SNRdB',20,'MaximumDopplerShift',100,'Seed',seed);
    ctx = rx.context(cfg);
    [rxWaveform,truth] = run_link(cfg);
    rFCNN = evaluate(nn.receive(rxWaveform,ctx,model),truth,cfg);
    rLS = evaluate(rx.receive(rxWaveform,ctx,'conventional'),truth,cfg);
    counts(1,:) = counts(1,:) + [rFCNN.UncodedErrors rFCNN.UncodedBits rFCNN.BlockError];
    counts(2,:) = counts(2,:) + [rLS.UncodedErrors rLS.UncodedBits rLS.BlockError];
end
ber = counts(:,1)./counts(:,2); bler = counts(:,3)/20;
passed = report(ber(1) < ber(2),sprintf('FCNN BER %.4f, BLER %.2f | conventional BER %.4f, BLER %.2f', ...
    ber(1),bler(1),ber(2),bler(2))) && passed;

if passed, fprintf('All neural-receiver checks passed.\n'); else, fprintf('Some neural-receiver checks FAILED.\n'); end
end

%% Local helpers
function [value,gradient] = lossProbe(s,t,m)
    value = nn.bce_loss(s,t,m)*1;
    gradient = dlgradient(value,s);
end

function [logits,g] = inputGradient(params,batch,ctx)
    logits = nn.fcnn_forward(params,batch,false,ctx);
    loss = nn.bce_loss(logits,batch.Labels,batch.Mask);
    g = dlgradient(loss,batch.Slots);
    logits = extractdata(logits);
end

function model = legacyModel(old)
    % network.m saved 'positive means bit 1' logits on real([Y,Xp,Hraw]),
    % imag([Y,Xp,Hraw]): the same as deeprx-v1 and nn.to_llr's convention.
    assert(strcmp(old.ScoreConvention,'positive means bit 1') && ...
        strcmp(old.FeatureOrder,'real([Y,Xp,Hraw]), imag([Y,Xp,Hraw])'));
    model.Net = old.Net;
    model.InputSize = old.InputSize;
    model.FeatureLayout = 'deeprx-v1';
end

function ok = report(ok,label)
    if ok, fprintf('  PASS  %s\n',label); else, fprintf('  FAIL  %s\n',label); end
end
