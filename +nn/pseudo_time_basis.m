function A = pseudo_time_basis(ctx, M)
%PSEUDO_TIME_BASIS Unitary IFFT from the occupied subcarriers to M samples.
%   A = nn.pseudo_time_basis(ctx, M) returns the M x F matrix that places
%   the F occupied subcarriers in their bins of an M-point IFFT (zeros in
%   the guard band) and transforms them, scaled to keep the energy:
%     x = A*X  is  sqrt(M)*ifft(X placed in its bins),   A'*A = I (F x F)
%   so A' is the matching FFT back to the occupied subcarriers.
%   M = ctx.Nfft gives the transmitter's own sample grid: for a transmit
%   grid column X, A*X/sqrt(Nfft) is that OFDM symbol's waveform without
%   its CP, the samples the PA distorted. The frequency-domain hybrid
%   (nn.flow_ftf) moves into and out of its pseudo-time domain with A.
F = ctx.NumSubcarriers;
assert(M >= F, 'The pseudo-time transform needs at least F = %d points, not %d.', F, M);
k = (0:F-1) - F/2;          % Subcarrier offset from the carrier centre, as rx.context.
n = (0:M-1).';
A = exp(2i*pi*n*k/M)/sqrt(M);
end
