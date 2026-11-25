function [xhat, nu] = equalize(Y, H, sigma2, ctx)
%EQUALIZE LMMSE equalization of the data resource elements, any number of layers.
%   [xhat, nu] = rx.equalize(Y, H, sigma2, ctx)
%   Y: F x N x Nr x B; H: F x N x Nr x B x L, the channel each layer's
%   symbols see (layer in dimension 5); sigma2: noise power per RE,
%   1 x 1 x 1 x B. Symbols have unit power.
%   Outputs keep only data REs, in transmit order: (L*D) x B with
%   D = numel(ctx.DataIndices), symbol L*(i-1)+l being layer l of data RE i
%   (the transmitter's layer mapping), so rx.demap_llr gives the LLRs in
%   transmit bit order. nu is the variance of the remaining noise (plus,
%   with several layers, inter-layer interference) on each xhat, as the
%   soft demapper needs it.
%
%   One layer (the paper's baseline, Pihlajasalo et al., IEEE TWC 2023):
%   one stream y = h*x + n, g = sum_r |H_r|^2,
%       w = h' / (g + sigma2),  x_lmmse = w*y = mu*x + noise,  mu = g/(g + sigma2)
%   LMMSE shrinks the symbol by mu; the soft demapper needs it unbiased:
%       xhat = x_lmmse / mu = sum_r conj(H_r) Y_r / g
%       nu   = (1 - mu)/mu   = sigma2 / g
%   (nu large where the channel fades). For one stream this is exactly
%   maximum-ratio combining, which is how it is computed. Products and
%   sums only: this path runs on dlarray.
%
%   Several layers (SU-MIMO): per RE, y = H x + n with H Nr x L. With
%   A = H'H + sigma2*I (L x L),
%       x_lmmse = A^-1 H' y,   E[x_lmmse | x] = (I - sigma2*A^-1) x
%   so layer l carries mu_l = 1 - sigma2*[A^-1]_ll times its own symbol.
%   Treating the other layers' leakage as Gaussian noise, the unbiased
%   per-layer estimate and its error variance are
%       xhat_l = x_lmmse_l / mu_l,   nu_l = (1 - mu_l)/mu_l
%                                         = sigma2*[A^-1]_ll / (1 - sigma2*[A^-1]_ll)
%   With one layer these are the formulas above. sigma2 = 0 gives zero
%   forcing (needs H of full column rank). Uses pageinv: numeric only.
L = size(H, 5);
assert(L == ctx.NumLayers, 'H has %d layers; ctx says %d.', L, ctx.NumLayers);
B = size(Y, 4);
FN = ctx.NumSubcarriers * ctx.NumSymbols;
if L == 1
    gain = sum(real(H).^2 + imag(H).^2, 3);  % F x N x 1 x B
    combined = sum(conj(H) .* Y, 3);
    gain = reshape(gain, FN, B);
    combined = reshape(combined, FN, B);
    gain = max(gain(ctx.DataIndices, :), eps);
    xhat = combined(ctx.DataIndices, :) ./ gain;
    nu = reshape(sigma2, 1, B) ./ gain;
    return
end

% Data REs as pages: Hd Nr x L x D x B, Yd Nr x 1 x D x B.
Nr = size(Y, 3);
D = numel(ctx.DataIndices);
H = reshape(H, FN, Nr, size(H, 4), L);
H = H(ctx.DataIndices, :, :, :);
if size(H, 3) < B, H = repmat(H, 1, 1, B, 1); end  % One channel for the whole batch.
Hd = permute(H, [2 4 1 3]);                         % Nr x L x D x B
Y = reshape(Y, FN, Nr, B);
Yd = permute(Y(ctx.DataIndices, :, :), [2 4 1 3]);  % Nr x 1 x D x B
s = reshape(sigma2, 1, 1, 1, B);

A = pagemtimes(Hd, 'ctranspose', Hd, 'none') + s .* eye(L);  % L x L x D x B
Ainv = pageinv(A);
z = pagemtimes(Ainv, pagemtimes(Hd, 'ctranspose', Yd, 'none'));  % L x 1 x D x B
Ainv = reshape(Ainv, L*L, D, B);
dA = real(Ainv(1:L+1:L*L, :, :));                   % L x D x B, [A^-1]_ll
e = reshape(s, 1, 1, B) .* dA;                      % sigma2*[A^-1]_ll = 1 - mu
mu = max(1 - e, eps);
xhat = reshape(reshape(z, L, D, B) ./ mu, L*D, B);  % Layer fastest: transmit order.
nu = reshape(e ./ mu, L*D, B);
end
