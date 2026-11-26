function passed = test_layers(action)
%% Check the layer-general link: SISO, SIMO and the SU-MIMO shapes
% Run: test_layers            (checks)
%      test_layers('update')  (store new reference digests, see part 1)
% 1. One layer is unchanged: a fixed set of SISO/SIMO links gives exactly
%    the outputs stored in test_layers_reference.mat (checksums of the
%    received samples, the truth, the conventional receiver's LLRs and the
%    network inputs and labels). The reference was stored from the code
%    before layers were added, so any difference means the single-layer
%    link changed. Checksums are exact: another MATLAB release or CPU can
%    change the last bit, so after checking that, store them again with
%    test_layers('update').
% 2. Shapes with one layer are the ones the saved models were trained on.
% 3. Two layers through the transmitter and the channel: shapes, the layer
%    mapping against nrLayerMap, total transmit power, and the received
%    grid against sum over layers of H .* X (the layout of truth.H).
% 4. Two layers through labels and LLR order, with LDPC: oracle LLRs decode.
% 5. Two layers with the known channel (stage 1): rx.equalize against a
%    per-RE reference, its one-layer limit, and the 'knownH' receiver end
%    to end (noiseless symbols, noise estimate, LDPC decoding).
% 6. Two layers from pilots (stage 2), DMRS ports 0 and 2 on separate
%    combs: per-layer pilot layout, LS estimates against the true channel,
%    the conventional receiver end to end, and the DMRSPortSet checks.
% 7. What is not implemented yet fails with a clear message.

if nargin < 1, action = 'check'; end
passed = true;
referenceFile = fullfile(fileparts(mfilename('fullpath')), 'test_layers_reference.mat');

%% 1. One layer: unchanged outputs
fprintf('1. One layer (SISO and SIMO): outputs against the stored reference\n');
cases = {
    'default',      {}
    'qpsk_ldpc',    {'Modulation','QPSK','Coding','ldpc','DMRSAdditionalPosition',1}
    'siso_64qam',   {'Modulation','64QAM','NumReceiveAntennas',1,'ChannelModel','TDL-A','SNRdB',25}
    'nr4_cdl',      {'NumReceiveAntennas',4,'ChannelModel','CDL-C','Coding','ldpc'}
    'interferer',   {'SIRdB',10,'Coding','ldpc','SNRdB',15}
    'pa_poly',      {'PAModel','polynomial','PABackoffdB',4,'SNRdB',20}
    'pa_mp3gpp',    {'PAModel','mp3gpp','PABackoffdB',2,'Coding','ldpc','SNRdB',25}
    'dmrs_type2',   {'DMRSConfigurationType',2,'NumCDMGroupsWithoutData',2,'DMRSLength',2,'DMRSAdditionalPosition',1}
    'perfect_time', {'TimingEstimation','perfect','MaximumDopplerShift',400,'Seed',7}
    };
digests = struct();
for k = 1:size(cases,1)
    digests.(cases{k,1}) = singleLayerDigests(link_config(cases{k,2}{:}));
end
if strcmp(action,'update')
    release = version; %#ok<NASGU>
    save(referenceFile,'digests','release');
    fprintf('  Stored %d cases in %s.\n',size(cases,1),referenceFile);
elseif ~isfile(referenceFile)
    passed = report(false,'no test_layers_reference.mat; run test_layers(''update'') on known-good code') && passed;
else
    ref = load(referenceFile);
    for k = 1:size(cases,1)
        name = cases{k,1};
        now = digests.(name); was = ref.digests.(name);
        changed = fieldnames(now);
        changed = changed(~cellfun(@(f) strcmp(now.(f),was.(f)),changed));
        label = sprintf('%-13s %d outputs identical',name,numel(fieldnames(now)));
        if ~isempty(changed)
            label = sprintf('%-13s changed: %s (reference from MATLAB %s)',name,strjoin(changed.',', '),ref.release);
        end
        passed = report(isempty(changed),label) && passed;
    end
end

%% 2. One layer: the shapes of the saved models
fprintf('2. One layer: shapes\n');
cfg = link_config('Coding','ldpc');
ctx = rx.context(cfg);
[~,truth] = sim.run_link(cfg);
F = ctx.NumSubcarriers; N = ctx.NumSymbols; Nr = cfg.NumReceiveAntennas; Q = ctx.BitsPerSymbol;
ok = isequal(size(truth.H),[F N Nr]) && isequal(size(truth.Grid),[F N]) && ...
    isequal(size(ctx.PilotGrid),[F N]) && iscolumn(ctx.PilotWaveform) && isscalar(truth.PAGain);
passed = report(ok,'truth.H is F x N x Nr; grid, pilot grid and pilot waveform have no layer dimension') && passed;
saved = load(fullfile(fileparts(mfilename('fullpath')),'models','fcnn.mat'),'model');
net = saved.model.Nets.fcnn;
planes = nn.labels(truth.CodedBits,ctx);
ok = isequal(net.Layers(1).InputSize,nn.fcnn_input_size(ctx)) && ...
    isequal(nn.fcnn_input_size(ctx),[F N 2*(2*Nr+1)]) && size(planes,3) == Q;
passed = report(ok,sprintf('models/fcnn.mat input %s equals nn.fcnn_input_size; %d bit planes', ...
    mat2str(net.Layers(1).InputSize),Q)) && passed;

%% 3. Two layers: transmitter and channel
fprintf('3. Two layers (2 Tx, 2 Rx): transmitter and channel\n');
two = {'NumTransmitAntennas',2,'NumLayers',2,'Modulation','QPSK'};
cfg = link_config(two{:},'SNRdB',Inf,'MaximumDopplerShift',0,'TimingEstimation','perfect');
ctx = rx.context(cfg);
[rxWaveform,truth,trace] = sim.run_link(cfg);
L = 2; Nt = 2; Nr = cfg.NumReceiveAntennas; D = numel(ctx.DataIndices);
ok = isequal(size(trace.Tx.Grid),[F N Nt]) && size(trace.Tx.WaveformBeforePA,2) == Nt && ...
    isequal(size(truth.H),[F N Nr 1 Nt]) && isequal(size(truth.PAGain),[1 1 1 1 Nt]) && ...
    isequal(size(ctx.PilotGrid),[F N 1 1 L]) && isequal(size(ctx.PilotWaveform),[ctx.SlotLength L]) && ...
    numel(truth.CodedBits) == 2*L*D;
passed = report(ok,'grid F x N x Nt, truth.H F x N x Nr x 1 x Nt, pilot grid F x N x 1 x 1 x L') && passed;

symbols = nrSymbolModulate(truth.CodedBits,cfg.Modulation);
layered = nrLayerMap(symbols,L);                       % D x L, TS 38.211 6.3.1.3.
sent = reshape(trace.Tx.Grid,F*N,Nt)*sqrt(Nt);
err = relErr(sent(ctx.DataIndices,:),layered);
passed = report(err < 1e-12,sprintf('layer mapping versus nrLayerMap: %.1e',err)) && passed;

one = link_config('Modulation','QPSK','SNRdB',Inf);
[~,~,traceOne] = sim.run_link(one);
power = @(w) sum(mean(abs(w).^2,1));
ratio = power(trace.Tx.WaveformBeforePA)/power(traceOne.Tx.WaveformBeforePA);
passed = report(abs(ratio-1) < 0.02,sprintf('total transmit power, two layers / one layer: %.4f',ratio)) && passed;

% Static channel, no noise, true timing: Y = sum over layers of H .* X.
Y = rx.ofdm_demod(rx.sync(rxWaveform,ctx,truth.TimingOffset),ctx);
X = reshape(trace.Tx.Grid,F,N,1,1,Nt);
err = relErr(Y,sum(truth.H .* X,5));
passed = report(err < 1e-2,sprintf('received grid versus sum over layers of H .* X: %.1e',err)) && passed;

worst = 0;
for seed = 1:10
    noisyCfg = link_config(two{:},'SNRdB',10,'Seed',seed);
    [noisy,noisyTruth] = sim.run_link(noisyCfg);
    [~,offset] = rx.sync(noisy,rx.context(noisyCfg));
    worst = max(worst,abs(offset - noisyTruth.TimingOffset));
end
passed = report(worst <= 4,sprintf( ...
    'rx.sync (dmrs) on both layers'' pilots, 10 slots at 10 dB: at most %d samples from the true timing',worst)) && passed;

%% 4. Two layers: labels, LLR order and LDPC
fprintf('4. Two layers: labels -> oracle logits -> LLRs -> LDPC decoding\n');
for modulation = {'QPSK','16QAM'}
    cfg = link_config(two{:},'Modulation',modulation{1},'Coding','ldpc','SNRdB',Inf);
    ctx = rx.context(cfg);
    [~,truth] = sim.run_link(cfg);
    Q = ctx.BitsPerSymbol;
    [planes,mask] = nn.labels(truth.CodedBits,ctx);
    llr = nn.to_llr(10*(2*planes-1),ctx);
    r = sim.evaluate(llr,truth,cfg);
    % Plane Q*(l-1)+b is bit b of layer l: check against the layered bits.
    bits = reshape(truth.CodedBits,Q,L,[]);                  % bit, layer, data RE
    flat = reshape(planes,[],Q*L);
    onGrid = reshape(permute(bits,[3 1 2]),[],Q*L);          % D x (Q*L), layer by layer
    ok = isequal(size(planes),[F N Q*L]) && isequal(flat(ctx.DataIndices,:),single(onGrid)) && ...
        nnz(mask) == numel(truth.CodedBits) && r.UncodedErrors == 0 && r.CodedErrors == 0 && r.BlockError == 0 && ...
        isequal(nn.to_planes(-llr,ctx),10*(2*planes-1).*mask) && ...
        isequal(nn.fcnn_input_size(ctx),[F N 2*(Nr+L+Nr*L)]);
    passed = report(ok,sprintf('%-5s %d bit planes, %d-bit transport block: 0 errors, CRC passes', ...
        modulation{1},Q*L,numel(truth.InfoBits))) && passed;
end
cfg = link_config(two{:},'SNRdB',20);
ctx = rx.context(cfg);
rxWaveform = sim.run_link(cfg);
Z = nn.deeprx_features(rx.ofdm_demod(rx.sync(rxWaveform,ctx),ctx),ctx.PilotGrid);
passed = report(isequal(size(Z),nn.fcnn_input_size(ctx)), ...
    sprintf('nn.deeprx_features gives %s',mat2str(size(Z)))) && passed;

%% 5. Two layers with the known channel: MIMO LMMSE detection
fprintf('5. Two layers: MIMO LMMSE detection with the known channel\n');
% rx.equalize against the textbook formulas, one RE at a time.
rng(1);
D = numel(ctx.DataIndices);
for Nr = [2 4]
    B = 2; sig = reshape([0.05 0.3],1,1,1,B);
    Hr = (randn(F,N,Nr,B,L) + 1i*randn(F,N,Nr,B,L))/sqrt(2);
    Yr = (randn(F,N,Nr,B) + 1i*randn(F,N,Nr,B))/sqrt(2);
    [xhat,nu] = rx.equalize(Yr,Hr,sig,ctx);
    xRef = zeros(L*D,B); nuRef = xRef;
    for b = 1:B
        for i = 1:D
            [f,n] = ind2sub([F N],ctx.DataIndices(i));
            h = reshape(Hr(f,n,:,b,:),Nr,L); y = reshape(Yr(f,n,:,b),Nr,1);
            Ai = inv(h'*h + sig(b)*eye(L));
            mu = real(diag(Ai*(h'*h)));
            xRef(L*(i-1)+(1:L),b) = (Ai*h'*y)./mu;
            nuRef(L*(i-1)+(1:L),b) = (1-mu)./mu;
        end
    end
    err = max(relErr(xhat,xRef),relErr(nu,nuRef));
    passed = report(err < 1e-10,sprintf('rx.equalize, %d x 2, batch of 2, versus per-RE inv(H''H+s*I)*H''*y: %.1e',Nr,err)) && passed;
end
% A layer whose channel is zero leaves the other layer's LMMSE = MRC.
Hz = Hr(:,:,:,:,1); Hz(:,:,:,:,2) = 0;
[xhat,nu] = rx.equalize(Yr,Hz,sig,ctx);
ctx1 = ctx; ctx1.NumLayers = 1;
[x1,nu1] = rx.equalize(Yr,Hz(:,:,:,:,1),sig,ctx1);
err = max(relErr(xhat(1:L:end,:),x1),relErr(nu(1:L:end,:),nu1));
passed = report(err < 1e-12,sprintf('second layer silent: first layer equals the one-layer equalizer: %.1e',err)) && passed;

% Noiseless static link: the detected symbols are the transmitted ones.
cfg = link_config(two{:},'Modulation','16QAM','SNRdB',Inf,'MaximumDopplerShift',0);
ctx = rx.context(cfg);
[rxWaveform,truth,trace] = sim.run_link(cfg);
[~,t] = rx.receive(rxWaveform,ctx,'knownH',truth);
% truth.H matches the channel to about 1e-3 (part 3), so this is an RMS
% check (EVM): the worst single RE is amplified by poorly conditioned H.
evm = rms(t.Xhat - trace.Tx.Symbols)/rms(trace.Tx.Symbols);
passed = report(evm < 1e-2 && t.Sigma2 < 1e-4,sprintf( ...
    'knownH, no noise, static: EVM %.1f dB, noise estimate %.1e',20*log10(evm),t.Sigma2)) && passed;

% Coded, 20 dB, TDL-C at 100 Hz: decodes, and the pilot noise estimate is close.
for modulation = {'QPSK','16QAM'}
    cfg = link_config(two{:},'Modulation',modulation{1},'Coding','ldpc','SNRdB',20);
    ctx = rx.context(cfg);
    [rxWaveform,truth] = sim.run_link(cfg);
    [llr,t] = rx.receive(rxWaveform,ctx,'knownH',truth);
    r = sim.evaluate(llr,truth,cfg);
    ratio = t.Sigma2/truth.NoiseVariance;
    passed = report(r.BlockError == 0 && ratio > 0.8 && ratio < 1.5,sprintf( ...
        '%-5s knownH at 20 dB: uncoded BER %.4f, CRC passes; noise estimate / true %.2f', ...
        modulation{1},r.UncodedErrors/r.UncodedBits,ratio)) && passed;
end

%% 6. Two layers from pilots: DMRS ports 0 and 2
fprintf('6. Two layers from pilots: DMRS ports 0 and 2 (separate combs)\n');
combs = {'DMRSConfigurationType',1,'DMRSPortSet',[0 2],'NumCDMGroupsWithoutData',2, ...
    'DMRSLength',1,'DMRSAdditionalPosition',0};
cfg = link_config(two{:},combs{:},'SNRdB',Inf,'MaximumDopplerShift',0,'TimingEstimation','perfect');
ctx = rx.context(cfg);
ok = ctx.PilotsSeparable && isequal(ctx.PilotRows,[(1:2:F).' (2:2:F).']) && ...
    numel(ctx.DataIndices) == F*(N-numel(ctx.PilotCols)) && ...
    ~rx.context(link_config(two{:})).PilotsSeparable;
passed = report(ok,sprintf(['layer 1 on the even comb, layer 2 on the odd comb, no data in the ' ...
    'pilot symbol (%d data REs); ports 0 and 1 are not separable'],numel(ctx.DataIndices))) && passed;
[rxWaveform,truth] = sim.run_link(cfg);
[~,t] = rx.receive(rxWaveform,ctx,'conventional',truth);
Ht = truth.H/sqrt(2);                                   % What the pilots see.
err = 0;
for l = 1:L
    err = max(err,relErr(t.Hp(:,:,:,1,l),Ht(ctx.PilotRows(:,l),ctx.PilotCols,:,1,l)));
end
interp = rms(t.H(:)-Ht(:))/rms(Ht(:));
% Floor: truth.H matches the channel to about 1e-3 (part 3).
passed = report(err < 5e-3 && interp < 1e-2 && t.Sigma2 < 1e-4,sprintf( ...
    'no noise, static: LS at each layer''s pilots within %.1e of H/sqrt(2), interpolated %.1e rms, noise %.1e', ...
    err,interp,t.Sigma2)) && passed;

for modulation = {'QPSK','16QAM'}
    cfg = link_config(two{:},combs{:},'Modulation',modulation{1},'Coding','ldpc','SNRdB',20);
    ctx = rx.context(cfg);
    [rxWaveform,truth] = sim.run_link(cfg);
    [llr,t] = rx.receive(rxWaveform,ctx,'conventional',truth);
    r = sim.evaluate(llr,truth,cfg);
    [~,tk] = rx.receive(rxWaveform,ctx,'knownH',truth);
    ratio = t.Sigma2/truth.NoiseVariance;
    passed = report(r.BlockError == 0 && ratio > 0.8 && ratio < 1.5 && ~isempty(tk.Hp) && tk.Sigma2 == t.Sigma2, ...
        sprintf(['%-5s conventional at 20 dB: uncoded BER %.4f, CRC passes; noise estimate / true %.2f ' ...
        '(knownH uses the same)'],modulation{1},r.UncodedErrors/r.UncodedBits,ratio)) && passed;
end

passed = report(fails(@() link_config(two{:},'DMRSPortSet',[0 2]),'NumCDMGroupsWithoutData'), ...
    'ports 0 and 2 with data in the second comb: link_config refuses') && passed;
passed = report(fails(@() link_config(two{:},'DMRSPortSet',0),'NumLayers different ports') && ...
    fails(@() link_config(two{:},'DMRSPortSet',[2 2],'NumCDMGroupsWithoutData',2),'NumLayers different ports'), ...
    'one port for two layers, or the same port twice: link_config refuses') && passed;
passed = report(fails(@() link_config(two{:},'DMRSPortSet',[0 4],'NumCDMGroupsWithoutData',2),'ports 0..3'), ...
    'port 4 with single-symbol type 1 DMRS: link_config refuses') && passed;

%% 7. Not implemented yet: clear errors
fprintf('7. Not implemented yet\n');
cfg = link_config(two{:},'SNRdB',20);                   % Default ports 0 and 1: one CDM group.
ctx = rx.context(cfg);
[rxWaveform,truth] = sim.run_link(cfg);
passed = report(fails(@() rx.receive(rxWaveform,ctx,'conventional'),'DMRS ports'), ...
    'conventional receiver with two layers: rx.ls_estimate refuses') && passed;
passed = report(fails(@() rx.receive(rxWaveform,ctx,'dpoi_knownH',truth),'one layer'), ...
    'DPoI with two layers: rx.dpoi refuses') && passed;
passed = report(fails(@() link_config('NumLayers',2),'NumTransmitAntennas'), ...
    'two layers from one transmit antenna: link_config refuses') && passed;
passed = report(fails(@() link_config('NumLayers',2,'NumTransmitAntennas',2,'NumReceiveAntennas',1),'NumReceiveAntennas'), ...
    'two layers to one receive antenna: link_config refuses') && passed;

if passed, fprintf('All layer checks passed.\n'); else, fprintf('Some layer checks FAILED.\n'); end
end

%% Local helpers
function d = singleLayerDigests(cfg)
    ctx = rx.context(cfg);
    [rxWaveform,truth] = sim.run_link(cfg);
    [llr,t] = rx.receive(rxWaveform,ctx,'conventional',truth);
    d.rxWaveform = digest(rxWaveform);
    d.CodedBits = digest(truth.CodedBits);
    d.Grid = digest(truth.Grid);
    d.H = digest(truth.H);
    d.PAGain = digest(truth.PAGain);
    d.PilotWaveform = digest(ctx.PilotWaveform);
    d.Offset = digest(t.Offset);
    d.Hest = digest(t.H);
    d.Sigma2 = digest(t.Sigma2);
    d.llrConventional = digest(llr);
    d.llrKnownH = digest(rx.receive(rxWaveform,ctx,'knownH',truth));
    if ~strcmp(cfg.PAModel,'none')
        d.llrDPoI = digest(rx.receive(rxWaveform,ctx,'dpoi',truth));
    end
    Y = rx.ofdm_demod(rx.sync(rxWaveform,ctx,t.Offset),ctx);
    d.Features = digest(nn.deeprx_features(Y,ctx.PilotGrid));
    [planes,mask] = nn.labels(truth.CodedBits,ctx);
    d.Labels = digest(planes);
    d.Mask = digest(mask);
    d.Planes = digest(nn.to_planes(-llr,ctx));
end

function h = digest(x)
    % Checksum of the size and the exact bits of the values (as doubles,
    % real then imaginary): two sums over their 32-bit words, one plain and
    % one weighted by position, modulo a prime. Any single changed bit
    % changes it. All arithmetic is exact in double precision.
    v = double(x(:));
    words = double(typecast([size(x).'; real(v); imag(v)],'uint32'));
    prime = 4294967291;
    weights = mod((1:numel(words)).',65521) + 1;
    h = sprintf('%010.0f-%010.0f',mod(sum(words),prime),mod(sum(mod(words.*weights,prime)),prime));
end

function ok = fails(f,fragment)
    % True when f() raises an error whose message contains fragment.
    ok = false;
    try
        f();
    catch err
        ok = contains(err.message,fragment);
    end
end

function e = relErr(a,b)
    e = max(abs(a(:)-b(:)))/max(abs(b(:)));
end

function ok = report(ok,label)
    if ok, fprintf('  PASS  %s\n',label); else, fprintf('  FAIL  %s\n',label); end
end
