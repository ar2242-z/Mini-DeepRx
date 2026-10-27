function passed = test_tx(referenceFile)
%% Check the transmitter (+tx), uncoded and with LDPC
% Run: test_tx
% 1. Uncoded: tx.transmit reproduces the legacy transmitter saved in
%    reference/ref_v0.mat (bits, grid, masks, PA gain) for every slot case.
% 2. LDPC: tx.transmit's coded bits equal MATLAB's nrULSCH bit for bit,
%    including a slot large enough for several code blocks.
% 3. Loopback: tx.transmit -> OFDM demodulation -> AWGN -> soft demap ->
%    rx.ldpc_decode. No errors without noise; BLER falls from 1 to 0 as SNR
%    rises, and coding beats the uncoded BER at the same SNR.
% Steps 2-3 use 5G Toolbox stand-ins (nrOFDMDemodulate, nrSymbolDemodulate)
% for receiver blocks that do not exist yet.

if nargin == 0
    referenceFile = fullfile(fileparts(mfilename('fullpath')),'reference','ref_v0.mat');
end
quiet = {'ShowPlots',false,'RunChecks',false,'SaveData',false,'Verbose',false};
passed = true;

%% 1. Uncoded transmitter against the legacy reference
fprintf('1. Uncoded Tx versus legacy reference (%s)\n',referenceFile);
saved = load(referenceFile,'ref');
for c = fieldnames(saved.ref.slots).'
    s = saved.ref.slots.(c{1});
    cfg = fromStruct(s.cfg);
    [~,out] = tx.transmit(cfg,RandStream('mt19937ar','Seed',cfg.Seed));
    ok = isequal(out.InfoBits,s.tx.Bits) && isequal(out.CodedBits,s.tx.Bits) && ...
        isequal(out.Grid,s.tx.X) && isequal(out.DataMask,s.tx.DataMask) && ...
        isequal(out.PilotMask,s.tx.PilotMask) && ...
        isequal(out.Resources.DataIndices,s.tx.DataIndices) && ...
        abs(out.PAGain-s.tx.PAGain) <= 1e-12*abs(s.tx.PAGain);
    passed = report(ok,c{1}) && passed;
end

%% 2. LDPC encoder against nrULSCH
fprintf('2. LDPC coded bits versus nrULSCH\n');
cases = {
    '16QAM R=0.50',             {}
    'QPSK R=0.30',              {'Modulation','QPSK','TargetCodeRate',0.3}
    '16QAM R=0.75 2 DMRS',      {'TargetCodeRate',0.75,'DMRSAdditionalPosition',1}
    '16QAM R=0.80 106 RB (C>1)',{'TargetCodeRate',0.8,'NumResourceBlocks',106,'Nfft',2048}
    };
for k = 1:size(cases,1)
    cfg = simo_config(quiet{:},'Coding','ldpc',cases{k,2}{:});
    [~,out] = tx.transmit(cfg,RandStream('mt19937ar','Seed',k));
    ulsch = nrULSCH('TargetCodeRate',cfg.TargetCodeRate);
    setTransportBlock(ulsch,out.InfoBits);
    reference = double(ulsch(cfg.Modulation,1,numel(out.CodedBits),0));
    ok = isequal(out.CodedBits,reference);
    passed = report(ok,sprintf('%-26s TBS %5d, G %5d, %d code block(s), rate %.3f', ...
        cases{k,1},out.Code.TBS,out.Code.G,out.Code.C,out.Code.EffectiveCodeRate)) && passed;
end

%% 3. Coded loopback over AWGN on the grid
fprintf('3. Loopback: Tx -> OFDM demod -> AWGN -> soft demap -> LDPC decode\n');
cfg = simo_config(quiet{:},'Coding','ldpc');
[waveform,out] = tx.transmit(cfg,RandStream('mt19937ar','Seed',1));
grid = nrOFDMDemodulate(out.Carrier,waveform,'Nfft',cfg.Nfft);
llr = nrSymbolDemodulate(grid(out.DataMask),cfg.Modulation,1e-3);
[tb,crcOK] = rx.ldpc_decode(llr,out.Code,cfg.LDPCMaxIterations);
passed = report(crcOK && isequal(tb,out.InfoBits) && ...
    max(abs(grid(:)-out.Grid(:))) < 1e-10,'noiseless: grid recovered, CRC passes, TB equal') && passed;

snrs = 0:2:12; numSlots = 20;
noiseStream = RandStream('mt19937ar','Seed',7);
bler = zeros(size(snrs)); codedBER = bler; uncodedBER = bler;
for i = 1:numel(snrs)
    N0 = 10^(-snrs(i)/10); % Es/N0 per data RE; symbols have unit energy.
    errors = 0; infoErrors = 0; rawErrors = 0;
    for slot = 1:numSlots
        [~,out] = tx.transmit(cfg,RandStream('mt19937ar','Seed',100+slot));
        x = out.Grid(out.DataMask);
        y = x + sqrt(N0/2)*(randn(noiseStream,size(x))+1i*randn(noiseStream,size(x)));
        llr = nrSymbolDemodulate(y,cfg.Modulation,N0);
        [tb,crcOK] = rx.ldpc_decode(llr,out.Code,cfg.LDPCMaxIterations);
        errors = errors + ~crcOK;
        infoErrors = infoErrors + nnz(tb~=out.InfoBits);
        rawErrors = rawErrors + nnz((llr<0)~=out.CodedBits); % Hard decision.
    end
    bler(i) = errors/numSlots;
    codedBER(i) = infoErrors/(numSlots*out.Code.TBS);
    uncodedBER(i) = rawErrors/(numSlots*out.Code.G);
end
disp(table(snrs(:),uncodedBER(:),codedBER(:),bler(:),'VariableNames', ...
    {'SNRdB','UncodedBER','CodedBER','BLER'}));
% Past the waterfall (BLER < 1) decoding must remove errors, not add them.
decoding = bler < 1;
ok = bler(1)==1 && bler(end)==0 && all(diff(bler)<=0.2) && any(decoding) && ...
    all(codedBER(decoding) < uncodedBER(decoding));
passed = report(ok,'BLER falls from 1 to 0; coded BER below uncoded wherever BLER < 1') && passed;

if passed, fprintf('All Tx checks passed.\n'); else, fprintf('Some Tx checks FAILED.\n'); end
end

%% Local helpers
function cfg = fromStruct(saved)
    % Rebuild a full configuration from a saved one; new fields get defaults.
    options = [fieldnames(saved).'; struct2cell(saved).'];
    cfg = simo_config(options{:});
end

function ok = report(ok,label)
    if ok, fprintf('  PASS  %s\n',label); else, fprintf('  FAIL  %s\n',label); end
end
