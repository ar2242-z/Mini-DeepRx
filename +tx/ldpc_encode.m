function [coded, code] = ldpc_encode(tb, G, cfg)
%LDPC_ENCODE 5G UL-SCH channel coding: transport block to G coded bits.
%   [coded, code] = tx.ldpc_encode(tb, G, cfg) applies TS 38.212 6.2:
%     1. CRC attach (24 bits, or 16 for blocks up to 3824 bits) so the
%        receiver can tell whether the decoded block is correct;
%     2. code block segmentation into LDPC-sized blocks (base graph 1 or 2,
%        chosen from the block size and code rate);
%     3. LDPC encoding, which adds structured parity bits;
%     4. rate matching (redundancy version 0): puncture or repeat, then
%        bit-interleave per symbol, so exactly G bits fill the data REs.
%   Data scrambling (TS 38.211 6.3.1.1) is not applied.
%   code describes the code so rx.ldpc_decode can invert every step.
%   tb is a column of 0/1; coded is a G x 1 double column of 0/1.
code = nrDLSCHInfo(numel(tb), cfg.TargetCodeRate); % Same rules as UL-SCH.
code.TBS = numel(tb);
code.G = G;
code.RV = 0;
code.Modulation = cfg.Modulation;
code.TargetCodeRate = cfg.TargetCodeRate;
code.EffectiveCodeRate = (code.TBS + code.L)/G; % Including the CRC.

tbWithCRC = nrCRCEncode(tb, code.CRC);
codeBlocks = nrCodeBlockSegmentLDPC(tbWithCRC, code.BGN);
encoded = nrLDPCEncode(codeBlocks, code.BGN);
coded = double(nrRateMatchLDPC(encoded, G, code.RV, cfg.Modulation, 1));
end
