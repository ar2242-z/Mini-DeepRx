function [waveform, out] = transmit(cfg, stream, cellID)
%TRANSMIT The whole transmitter for one slot: bits to PA output.
%   [waveform, out] = tx.transmit(cfg, stream) runs
%     bits -> [LDPC] -> symbol mapping -> grid with DMRS -> IFFT + CP -> PA
%   drawing bits from stream. cellID (default 0) sets the DMRS scrambling.
%   waveform is the PA output, a column of complex baseband samples.
%   out holds every intermediate result (the Tx checkpoints):
%     Carrier, Resources   numerology and data/DMRS positions
%     InfoBits             transport block (uncoded: equals CodedBits)
%     CodedBits            bits on the grid; label of each data RE
%     Code                 LDPC description for rx.ldpc_decode ([] if none)
%     Symbols, Grid        data symbols and the full resource grid
%     DataMask, PilotMask  logical masks of the grid
%     WaveformBeforePA     OFDM waveform before the PA
%     PAGain               Bussgang gain of the PA (1 when linear): a scalar
%                          for the memoryless miniPA, F x 1 (one gain per
%                          subcarrier) for the memory polynomial 'mp3gpp'
if nargin < 3
    cellID = 0;
end
carrier = tx.carrier(cfg, cellID);
res = tx.resources(carrier, cfg);
q = tx.bits_per_symbol(cfg.Modulation);
G = q*numel(res.DataIndices); % Coded bits that fit the data REs.

switch cfg.Coding
    case 'none'
        infoBits = tx.bits(G, stream);
        codedBits = infoBits;
        code = [];
    case 'ldpc'
        A = tx.transport_block_size(cfg, numel(res.DataIndices));
        infoBits = tx.bits(A, stream);
        [codedBits, code] = tx.ldpc_encode(infoBits, G, cfg);
end
symbols = tx.map_symbols(codedBits, cfg.Modulation);
[grid, dataMask, pilotMask] = tx.build_grid(carrier, res, symbols);
beforePA = tx.ofdm_mod(carrier, grid, cfg.Nfft);
if strcmp(cfg.PAModel, 'mp3gpp')
    [waveform, paGain] = tx.mppa(beforePA, carrier, cfg);
else
    [waveform, paGain] = tx.miniPA(beforePA, cfg);
end

out.Carrier = carrier;
out.Resources = res;
out.InfoBits = infoBits;
out.CodedBits = codedBits;
out.Code = code;
out.Symbols = symbols;
out.Grid = grid;
out.DataMask = dataMask;
out.PilotMask = pilotMask;
out.WaveformBeforePA = beforePA;
out.PAGain = paGain;
end
