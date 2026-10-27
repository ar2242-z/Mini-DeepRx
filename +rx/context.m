function ctx = context(cfg)
%CONTEXT Everything the receiver knows before a slot arrives.
%   ctx = rx.context(cfg) is built from the configuration only: numerology,
%   where data and pilots sit, the pilot values, and fixed operators for
%   the FFT window and pilot interpolation. It holds nothing about the
%   channel, the noise, or the data; a practical receiver uses only ctx and
%   the received waveform. Build it once and reuse it for every slot.
%   Fields:
%     Carrier, Nfft, NumSubcarriers (F), NumSymbols (N), NumReceiveAntennas
%     CyclicPrefixLengths, SlotLength      samples per slot including CPs
%     WindowIndices   Nfft x N: sample indices of each symbol's FFT window,
%                     starting halfway into its CP
%     ActiveBins      F x 1: FFT bins of the occupied subcarriers
%     PhaseCorrection F x N: undoes the early FFT window start
%     DataIndices, PilotIndices, PilotGrid, DataMask, PilotMask
%     PilotRows (P), PilotCols (K), PilotValues P x K
%     FrequencyInterp F x P, TimeInterp N x K   linear interpolation
%     SecondDifference  rows [1 -2 1] over equally spaced pilots
%     PilotWaveform   pilot-only transmit waveform, the sync reference
%     Modulation, BitsPerSymbol, LDPCMaxIterations, TimingEstimation
carrier = tx.carrier(cfg);
ofdm = nrOFDMInfo(carrier, 'Nfft', cfg.Nfft);
F = 12*cfg.NumResourceBlocks;
N = carrier.SymbolsPerSlot;
Nfft = cfg.Nfft;
cp = double(ofdm.CyclicPrefixLengths(1:N));

ctx.Carrier = carrier;
ctx.Nfft = Nfft;
ctx.NumSubcarriers = F;
ctx.NumSymbols = N;
ctx.NumReceiveAntennas = cfg.NumReceiveAntennas;
ctx.CyclicPrefixLengths = cp;
ctx.SlotLength = sum(cp) + N*Nfft;

% FFT window: start fix(cp/2) samples into the CP, so a small timing error
% or an early path still falls inside the CP (same as nrOFDMDemodulate).
% Starting `early` samples before the CP end rotates subcarrier k by
% exp(-2i*pi*k*early/Nfft); PhaseCorrection undoes it.
starts = cumsum([0 cp(1:end-1)+Nfft]); % First sample (CP) of each symbol, 0-based.
skip = fix(cp/2);
early = cp - skip;
ctx.WindowIndices = starts + skip + (1:Nfft).';
k = (0:F-1).' - F/2; % Subcarrier offset from the carrier centre.
ctx.ActiveBins = mod(k, Nfft) + 1;
ctx.PhaseCorrection = exp(2i*pi*k*early/Nfft);

% Data and pilot layout, and the known pilot values.
res = tx.resources(carrier, cfg);
ctx.DataIndices = res.DataIndices;
ctx.PilotIndices = res.PilotIndices;
ctx.PilotGrid = complex(zeros(F, N));
ctx.PilotGrid(res.PilotIndices) = res.Pilots;
ctx.DataMask = false(F, N);
ctx.DataMask(res.DataIndices) = true;
ctx.PilotMask = false(F, N);
ctx.PilotMask(res.PilotIndices) = true;
cols = find(any(ctx.PilotMask, 1));
rows = find(ctx.PilotMask(:, cols(1)));
assert(isequal(ctx.PilotMask(:, cols), repmat(ctx.PilotMask(:, cols(1)), 1, numel(cols))), ...
    'All pilot symbols must use the same subcarriers.');
ctx.PilotRows = rows;
ctx.PilotCols = cols;
ctx.PilotValues = ctx.PilotGrid(rows, cols);
assert(all(abs(abs(ctx.PilotValues(:)) - 1) < 1e-10), 'The receiver expects unit-power pilots.');

% Interpolation from pilots to the grid, as matrices (the paper's
% DMRS-based baseline):
%   H = FrequencyInterp * Hpilots * TimeInterp.'
% Frequency: linear, extrapolated at the band edges.
% Time: linear between pilot symbols, nearest pilot symbol held outside them.
P = numel(rows); K = numel(cols);
ctx.FrequencyInterp = interp1(rows, eye(P), (1:F).', 'linear', 'extrap');
if K == 1
    ctx.TimeInterp = ones(N, 1);
else
    t = min(max((1:N).', cols(1)), cols(end));
    ctx.TimeInterp = interp1(cols, eye(K), t, 'linear');
end

% Noise estimation: second differences over equally spaced pilots cancel a
% locally linear channel; with independent noise E|d|^2 = 6*sigma^2.
stride = pilotStride(rows);
D = zeros(0, P);
for o = 1:stride
    idx = o:stride:P;
    for j = 1:numel(idx)-2
        D(end+1, idx(j:j+2)) = [1 -2 1]; %#ok<AGROW>
    end
end
assert(~isempty(D), 'Noise estimation needs at least three equally spaced pilots.');
ctx.SecondDifference = D;

ctx.PilotWaveform = tx.ofdm_mod(carrier, ctx.PilotGrid, Nfft);
ctx.Modulation = cfg.Modulation;
ctx.BitsPerSymbol = tx.bits_per_symbol(cfg.Modulation);
ctx.LDPCMaxIterations = cfg.LDPCMaxIterations;
ctx.TimingEstimation = cfg.TimingEstimation;
end

function stride = pilotStride(f)
    % Smallest s such that f(o:s:end) is equally spaced for every offset o.
    % Type 1 DMRS: 1 (every 2nd subcarrier). Type 2: 2 (pairs every 6).
    for stride = 1:numel(f)
        uniform = true;
        for o = 1:stride
            spacing = diff(f(o:stride:end));
            uniform = uniform && all(spacing == spacing(1:min(1,end)));
        end
        if uniform, return; end
    end
end
