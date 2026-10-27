function [tb, crcOK, info] = ldpc_decode(llr, code, maxIterations)
%LDPC_DECODE Invert tx.ldpc_encode: G soft bits to the transport block.
%   [tb, crcOK] = rx.ldpc_decode(llr, code, maxIterations)
%   llr: G x 1 log-likelihood ratios, LLR = log P(b=0)/P(b=1), so positive
%   means bit 0 (nrSymbolDemodulate convention). code: from tx.ldpc_encode.
%   Steps: rate recovery (de-interleave, undo puncturing/repetition),
%   belief-propagation LDPC decoding, desegmentation, CRC check.
%   tb is a column of 0/1; crcOK is true when the transport block CRC passes.
%   info.CodeBlockCRCOK gives the per-code-block CRC (multi-block only).
assert(numel(llr) == code.G, 'Expected %d LLRs, got %d.', code.G, numel(llr));
recovered = nrRateRecoverLDPC(double(llr(:)), code.TBS, code.TargetCodeRate, ...
    code.RV, code.Modulation, 1);
decoded = nrLDPCDecode(recovered, code.BGN, maxIterations);
[blk, blockErrors] = nrCodeBlockDesegmentLDPC(decoded, code.BGN, code.TBS + code.L);
[tb, remainder] = nrCRCDecode(blk, code.CRC);
tb = double(tb);
crcOK = remainder == 0;
info.CodeBlockCRCOK = ~blockErrors;
end
