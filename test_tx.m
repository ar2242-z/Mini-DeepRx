function passed = test_tx()
%% Check the transmitter (+tx), uncoded and with LDPC
% Run: test_tx
% 1. LDPC: tx.transmit's coded bits equal MATLAB's nrULSCH bit for bit,
%    including a slot large enough for several code blocks.
% 2. Loopback: tx.transmit -> OFDM demodulation -> AWGN -> soft demap ->
%    rx.ldpc_decode. No errors without noise; BLER falls from 1 to 0 as SNR
%    rises, and coding beats the uncoded BER at the same SNR.
% Step 2 uses 5G Toolbox stand-ins (nrOFDMDemodulate, nrSymbolDemodulate)
% for receiver blocks that do not exist yet.

passed = true;

%% 1. LDPC encoder against nrULSCH
fprintf('1. LDPC coded bits versus nrULSCH\n');
cases = {
    '16QAM R=0.50',             {}
    'QPSK R=0.30',              {'Modulation','QPSK','TargetCodeRate',0.3}
    '16QAM R=0.75 2 DMRS',      {'TargetCodeRate',0.75,'DMRSAdditionalPosition',1}
    '16QAM R=0.80 106 RB (C>1)',{'TargetCodeRate',0.8,'NumResourceBlocks',106,'Nfft',2048}
    };
for k = 1:size(cases,1)
    cfg = simo_config('Coding','ldpc',cases{k,2}{:});
    [~,out] = tx.transmit(cfg,RandStream('mt19937ar','Seed',k));
    ulsch = nrULSCH('TargetCodeRate',cfg.TargetCodeRate);
    setTransportBlock(ulsch,out.InfoBits);
    reference = double(ulsch(cfg.Modulation,1,numel(out.CodedBits),0));
    ok = isequal(out.CodedBits,reference);
    passed = report(ok,sprintf('%-26s TBS %5d, G %5d, %d code block(s), rate %.3f', ...
        cases{k,1},out.Code.TBS,out.Code.G,out.Code.C,out.Code.EffectiveCodeRate)) && passed;
end

%% 2. Coded loopback over AWGN on the grid
fprintf('2. Loopback: Tx -> OFDM demod -> AWGN -> soft demap -> LDPC decode\n');
cfg = simo_config('Coding','ldpc');
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
function ok = report(ok,label)
    if ok, fprintf('  PASS  %s\n',label); else, fprintf('  FAIL  %s\n',label); end
end
