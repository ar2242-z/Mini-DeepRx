function result = evaluate(llr, truth, cfg)
%EVALUATE Error counts for one slot from the receiver's LLRs.
%   result = evaluate(llr, truth, cfg), llr from any receiver (log P(0)/P(1),
%   transmit bit order), truth from run_link.
%   Uncoded: hard decisions (llr < 0 means bit 1) against the bits on the
%   grid. Coded (cfg.Coding 'ldpc'): LDPC decoding, then the transport block
%   against the information bits, and the CRC for block errors.
%   Fields: UncodedErrors, UncodedBits, and when coded also
%   CodedErrors, CodedBits, BlockError (1 when the CRC fails).
llr = double(gather(extractdataIfNeeded(llr)));
result.UncodedErrors = nnz((llr < 0) ~= truth.CodedBits);
result.UncodedBits = numel(truth.CodedBits);
if strcmp(cfg.Coding, 'ldpc')
    [tb, crcOK] = rx.ldpc_decode(llr, truth.Code, cfg.LDPCMaxIterations);
    result.CodedErrors = nnz(tb ~= truth.InfoBits);
    result.CodedBits = numel(truth.InfoBits);
    result.BlockError = double(~crcOK);
end
end

function x = extractdataIfNeeded(x)
    if isa(x, 'dlarray'), x = extractdata(x); end
end
