function passed = test_rx(referenceFile)
%% Check the conventional receiver (+rx), block by block and end to end
% Run: test_rx
% 1. Against the earlier receiver (reference/ref_v0.mat): the saved Y fed
%    through ls_estimate -> interpolate -> equalize -> demap_llr reproduces
%    decode_simo_ls (H estimate, noise power, equalized symbols, hard bits)
%    for the estimated and the known H. rx.ofdm_demod equals nrOFDMDemodulate.
% 2. rx.demap_llr equals the 5G Toolbox soft demapper (nrSymbolDemodulate).
% 3. rx.sync ('dmrs') lands close to the true timing and costs little BER.
% 4. Every differentiable block gives the same result on dlarray, gradients
%    reach the received samples, and a batch of slots equals single slots.
% 5. End to end with LDPC: run_link -> rx.receive -> evaluate, SNR sweep,
%    the paper's two LMMSE baselines on identical received signals: LS +
%    linear interpolation, and known H.

if nargin == 0
    referenceFile = fullfile(fileparts(mfilename('fullpath')),'reference','ref_v0.mat');
end
quiet = {'ShowPlots',false,'RunChecks',false,'SaveData',false,'Verbose',false};
passed = true;

%% 1. Against the earlier receiver
fprintf('1. Receiver blocks versus legacy decode_simo_ls (%s)\n',referenceFile);
saved = load(referenceFile,'ref');
for c = fieldnames(saved.ref.slots).'
    s = saved.ref.slots.(c{1});
    ctx = rx.context(fromStruct(s.cfg));
    [Hp,sigma2] = rx.ls_estimate(s.rx.Y,ctx);
    H = rx.interpolate(Hp,ctx);
    [xLS,nuLS] = rx.equalize(s.rx.Y,H,sigma2,ctx);
    [xKnown,nuKnown] = rx.equalize(s.rx.Y,s.chan.H,sigma2,ctx);
    bitsLS = double(rx.demap_llr(xLS,nuLS,ctx.Modulation) < 0);
    bitsKnown = double(rx.demap_llr(xKnown,nuKnown,ctx.Modulation) < 0);
    errs = [relErr(H,s.ls.Hls), abs(sigma2/s.ls.sigma2Hat-1), ...
        relErr(xLS,s.ls.XhatLS(ctx.DataIndices)), relErr(xKnown,s.ls.XhatKnown(ctx.DataIndices))];
    ok = all(errs < 1e-10) && isequal(bitsLS,s.ls.bitsLS) && isequal(bitsKnown,s.ls.bitsKnown);
    passed = report(ok,sprintf('%-15s H %.0e, sigma2 %.0e, xhat LS %.0e, known %.0e; hard bits equal: %d/%d', ...
        c{1},errs,isequal(bitsLS,s.ls.bitsLS),isequal(bitsKnown,s.ls.bitsKnown))) && passed;
end
cfg = simo_config(quiet{:},'TimingEstimation','perfect');
ctx = rx.context(cfg);
[rxWaveform,truth] = run_link(cfg);
slot = rx.sync(rxWaveform,ctx,truth.TimingOffset);
err = relErr(rx.ofdm_demod(slot,ctx),nrOFDMDemodulate(ctx.Carrier,slot,'Nfft',cfg.Nfft));
passed = report(err < 1e-12,sprintf('rx.ofdm_demod versus nrOFDMDemodulate: %.1e',err)) && passed;

%% 2. Soft demapper against nrSymbolDemodulate
fprintf('2. rx.demap_llr versus nrSymbolDemodulate (max-log)\n');
stream = RandStream('mt19937ar','Seed',11);
for modulation = {'QPSK','16QAM'}
    q = tx.bits_per_symbol(modulation{1});
    x = nrSymbolModulate(randi(stream,[0 1],q*2000,1),modulation{1});
    nVar = 0.2;
    y = x + sqrt(nVar/2)*(randn(stream,size(x))+1i*randn(stream,size(x)));
    mine = rx.demap_llr(y,nVar*ones(size(y)),modulation{1});
    toolbox = nrSymbolDemodulate(y,modulation{1},nVar);
    err = relErr(mine,toolbox);
    passed = report(err < 1e-10,sprintf('%-6s relative difference %.1e',modulation{1},err)) && passed;
end

%% 3. Timing synchronization from the DMRS
fprintf('3. rx.sync (dmrs) versus true timing, 10 slots per case\n');
models = {'TDL-A','TDL-C','TDL-E','CDL-A','CDL-D'};
snrs = [0 10 30];
margin = fix(min(ctx.CyclicPrefixLengths)/2); % FFT window starts this far into the CP.
allErrors = [];
for m = models
    for snr = snrs
        e = zeros(10,1); berEst = 0; berTrue = 0;
        for seed = 1:10
            cfg = simo_config(quiet{:},'ChannelModel',m{1},'SNRdB',snr,'Seed',seed);
            [rxWaveform,truth] = run_link(cfg);
            ctxEst = rx.context(cfg);
            ctxTrue = ctxEst; ctxTrue.TimingEstimation = 'perfect';
            [llrEst,tr] = rx.receive(rxWaveform,ctxEst,'conventional');
            llrTrue = rx.receive(rxWaveform,ctxTrue,'conventional',truth);
            e(seed) = tr.Offset - truth.TimingOffset;
            berEst = berEst + mean((llrEst<0) ~= truth.CodedBits)/10;
            berTrue = berTrue + mean((llrTrue<0) ~= truth.CodedBits)/10;
        end
        allErrors = [allErrors; e]; %#ok<AGROW>
        ok = all(abs(e) <= margin) && berEst <= berTrue*1.1 + 1e-3;
        passed = report(ok,sprintf('%s %2d dB: timing error %s samples; BER %.4f (true timing %.4f)', ...
            m{1},snr,rangeText(e),berEst,berTrue)) && passed;
    end
end
fprintf('        all cases: %d%% exact, max |error| %d samples (FFT window margin %d)\n', ...
    round(100*mean(allErrors==0)),max(abs(allErrors)),margin);

%% 4. dlarray, gradients, and batches
fprintf('4. Differentiable blocks on dlarray, and batches\n');
cfg = simo_config(quiet{:},'SNRdB',15);
ctx = rx.context(cfg);
[rxWaveform,truth] = run_link(cfg);
slot = rx.sync(rxWaveform,ctx);
llr = chain(slot,ctx);
[llrDl,grad] = dlfeval(@chainLoss,dlarray(slot),ctx);
err = relErr(extractdata(llrDl),llr);
g = extractdata(grad);
passed = report(err < 1e-12 && all(isfinite(g(:))) && any(g(:)),sprintf( ...
    'dlarray LLRs equal double (%.1e); gradient w.r.t. samples finite and non-zero',err)) && passed;
cfg2 = cfg; cfg2.Seed = 7;
slot2 = rx.sync(run_link(cfg2),ctx);
llrBatch = chain(cat(3,slot,slot2),ctx);
err = max(relErr(llrBatch(:,1),llr),relErr(llrBatch(:,2),chain(slot2,ctx)));
passed = report(err < 1e-12,sprintf('batch of 2 slots equals single slots (%.1e)',err)) && passed;

%% 5. End to end with LDPC
fprintf('5. End to end: TDL-C 300 ns, 100 Hz, 16QAM, LDPC R=0.5, DMRS timing\n');
snrs = 0:4:28; numSlots = 30;
modes = {'knownH','conventional'};
ber = zeros(numel(snrs),2); bler = ber;
for i = 1:numel(snrs)
    counts = zeros(2,3);
    for seed = 1:numSlots
        cfg = simo_config(quiet{:},'Coding','ldpc','SNRdB',snrs(i),'Seed',1000+seed);
        ctx = rx.context(cfg);
        [rxWaveform,truth] = run_link(cfg);
        for m = 1:2
            r = evaluate(rx.receive(rxWaveform,ctx,modes{m},truth),truth,cfg);
            counts(m,:) = counts(m,:) + [r.UncodedErrors r.UncodedBits r.BlockError];
        end
    end
    ber(i,:) = counts(:,1)./counts(:,2);
    bler(i,:) = counts(:,3)/numSlots;
end
disp(table(snrs(:),ber(:,1),ber(:,2),bler(:,1),bler(:,2),'VariableNames', ...
    {'SNRdB','UncodedBER_knownH','UncodedBER_LS','BLER_knownH','BLER_LS'}));
ok = bler(1,2)==1 && bler(end,1)==0 && bler(end,2) <= 0.1 && ...
    all(bler(:,1) <= bler(:,2) + 2/numSlots) && all(ber(:,1) <= ber(:,2)) && ...
    all(diff(ber(:,2)) < 0);
passed = report(ok,['BLER 1 -> ~0; known H never worse than LS estimation; ' ...
    'uncoded BER falls with SNR']) && passed;

if passed, fprintf('All receiver checks passed.\n'); else, fprintf('Some receiver checks FAILED.\n'); end
end

%% Local helpers
function llr = chain(slot,ctx)
    % The differentiable part of the conventional receiver.
    Y = rx.ofdm_demod(slot,ctx);
    [Hp,sigma2] = rx.ls_estimate(Y,ctx);
    [xhat,nu] = rx.equalize(Y,rx.interpolate(Hp,ctx),sigma2,ctx);
    llr = rx.demap_llr(xhat,nu,ctx.Modulation);
end

function [llr,grad] = chainLoss(slot,ctx)
    llr = chain(slot,ctx);
    loss = sum(llr.^2,'all')/numel(llr);
    grad = dlgradient(loss,slot);
end

function t = rangeText(e)
    t = sprintf('%+d..%+d',min(e),max(e));
end

function e = relErr(a,b)
    e = norm(a(:)-b(:))/max(norm(b(:)),eps);
end

function cfg = fromStruct(saved)
    options = [fieldnames(saved).'; struct2cell(saved).'];
    cfg = simo_config(options{:});
end

function ok = report(ok,label)
    if ok, fprintf('  PASS  %s\n',label); else, fprintf('  FAIL  %s\n',label); end
end
