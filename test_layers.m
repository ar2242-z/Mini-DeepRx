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
% 5. What is not implemented yet fails with a clear message.

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

%% 5. Not implemented yet: clear errors
fprintf('5. Not implemented yet\n');
passed = report(fails(@() rx.receive(rxWaveform,ctx,'conventional'),'DMRS ports'), ...
    'conventional receiver with two layers: rx.ls_estimate refuses') && passed;
passed = report(fails(@() rx.equalize(zeros(F,N,2),zeros(F,N,2,1,2),1,ctx),'one layer'), ...
    'rx.equalize with two layers refuses') && passed;
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
