function Z = pair_transform(Z, T)
%PAIR_TRANSFORM Parallel linear transforms on real/imaginary channel pairs.
%   Z = nn.pair_transform(Z, T)
%   Z: R x N x C x B real (unformatted dlarray or numeric), C even. Channels
%   1..C/2 are the real parts and C/2+1..C the imaginary parts of C/2
%   complex signals, the layout of nn.deeprx_features. Each signal and OFDM
%   symbol (column) is multiplied by the complex matrix T (P x R), and the
%   result is split back into real and imaginary parts: P x N x C x B.
%   With T = nn.pseudo_time_basis(ctx, M) these are the paper's parallel
%   IFFTs into the pseudo-time domain; T' is the parallel FFTs back. All
%   C channels pass through, so the transform is no bottleneck (Pihlajasalo
%   et al., TWC 2023, Fig. 3). Products only: gradients pass through.
[R, N, C, B] = size(Z, 1:4);
assert(mod(C, 2) == 0, 'pair_transform needs an even number of channels, not %d.', C);
assert(size(T, 2) == R, 'T has %d columns; Z has %d rows.', size(T, 2), R);
h = C/2;
% real() on the inputs keeps the gradient flowing back into the network
% before real-valued, as its convolutions require.
c = real(Z(:, :, 1:h, :)) + 1i*real(Z(:, :, h+1:end, :));
x = reshape(T*reshape(c, R, []), size(T, 1), N, h, B);
Z = cat(3, real(x), imag(x));
end
