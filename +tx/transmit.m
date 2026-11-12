function [waveform, out] = transmit(cfg, stream, cellID)
%TRANSMIT The whole transmitter for one slot: bits to PA output.
%   [waveform, out] = tx.transmit(cfg, stream) runs
%     bits -> [LDPC] -> symbol mapping -> layer mapping -> grid with DMRS
%     -> IFFT + CP -> PA
%   drawing bits from stream. cellID (default 0) sets the DMRS scrambling.
%   waveform is the PA output: complex baseband samples, one column per
%   transmit antenna (Nt = cfg.NumTransmitAntennas).
%   Layers (L = cfg.NumLayers): one transport block is mapped across the
%   layers, symbol L*(i-1)+l on layer l of data RE i (TS 38.211 6.3.1.3).
%   There is no precoder: layer l is sent from antenna l. The whole grid,
%   pilots included, is scaled by 1/sqrt(Nt), so the total transmit power
%   does not depend on Nt and cfg.SNRdB keeps its meaning.
%   out holds every intermediate result (the Tx checkpoints):
%     Carrier, Resources   numerology and data/DMRS positions
%     InfoBits             transport block (uncoded: equals CodedBits)
%     CodedBits            bits on the grid, (Q*L*D) x 1; the Q bits of
%                          symbol 1, then of symbol 2, ...
%     Code                 LDPC description for rx.ldpc_decode ([] if none)
%     Symbols              data symbols, (L*D) x 1 in transmit order
%     Grid                 the transmitted resource grid, F x N x Nt
%     DataMask, PilotMask  logical masks of the grid
%     WaveformBeforePA     OFDM waveform before the PA, one column per antenna
%     PAGain               Bussgang gain of each antenna's PA (1 when
%                          linear) along dimension 5: 1 x 1 x 1 x 1 x Nt for
%                          the memoryless miniPA, F x 1 x 1 x 1 x Nt (one
%                          gain per subcarrier) for the memory polynomial
%                          'mp3gpp'. One antenna: a scalar or F x 1.
if nargin < 3
    cellID = 0;
end
carrier = tx.carrier(cfg, cellID);
res = tx.resources(carrier, cfg);
q = tx.bits_per_symbol(cfg.Modulation);
[D, L] = size(res.DataIndices);
Nt = cfg.NumTransmitAntennas;
G = q*D*L; % Coded bits that fit the data REs of all layers.

switch cfg.Coding
    case 'none'
        infoBits = tx.bits(G, stream);
        codedBits = infoBits;
        code = [];
    case 'ldpc'
        A = tx.transport_block_size(cfg, D);
        infoBits = tx.bits(A, stream);
        [codedBits, code] = tx.ldpc_encode(infoBits, G, cfg);
end
symbols = tx.map_symbols(codedBits, cfg.Modulation);
layered = reshape(symbols, L, D).'; % D x L: symbol L*(i-1)+l on layer l.
[grid, dataMask, pilotMask] = tx.build_grid(carrier, res, layered);
grid = grid / sqrt(Nt);             % Layer l on antenna l; fixed total power.
beforePA = tx.ofdm_mod(carrier, grid, cfg.Nfft);
% Each transmit antenna has its own PA (the same model and back-off).
waveform = complex(zeros(size(beforePA)));
gains = cell(1, 1, 1, 1, Nt);
for t = 1:Nt
    if strcmp(cfg.PAModel, 'mp3gpp')
        [waveform(:, t), gains{t}] = tx.mppa(beforePA(:, t), carrier, cfg);
    else
        [waveform(:, t), gains{t}] = tx.miniPA(beforePA(:, t), cfg);
    end
end
paGain = cat(5, gains{:});

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
