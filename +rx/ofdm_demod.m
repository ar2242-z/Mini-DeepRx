function Y = ofdm_demod(slot, ctx)
%OFDM_DEMOD CP removal and FFT: slot samples to the received grid.
%   Y = rx.ofdm_demod(slot, ctx) takes ctx.SlotLength x Nr x B samples
%   (from rx.sync; B slots) and returns Y, F x N x Nr x B: one complex value
%   per subcarrier, OFDM symbol and receive antenna.
%   Per symbol: keep the Nfft samples of its FFT window (this drops the CP),
%   FFT, keep the occupied bins, and undo the phase rotation of the early
%   window start. The FFT is unscaled, the inverse of tx.ofdm_mod.
%   Uses only indexing, fft and products, so it also runs on dlarray.
[~, Nr, B] = size(slot);
Nfft = ctx.Nfft; N = ctx.NumSymbols;
x = reshape(slot(ctx.WindowIndices(:), :, :), Nfft, N, Nr, B); % Drop CPs.
X = fft(x, [], 1);
Y = X(ctx.ActiveBins, :, :, :) .* ctx.PhaseCorrection;
end
