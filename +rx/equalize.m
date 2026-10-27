function [xhat, nu] = equalize(Y, H, sigma2, ctx)
%EQUALIZE LMMSE equalization of the data resource elements (SIMO).
%   [xhat, nu] = rx.equalize(Y, H, sigma2, ctx)
%   Y, H: F x N x Nr x B; sigma2: noise power per RE, 1 x 1 x 1 x B.
%   The LMMSE equalizer of the paper's baseline (Pihlajasalo et al., IEEE
%   TWC 2023), for one stream y = h*x + n with unit-power symbols and
%   g = sum_r |H_r|^2:
%       w = h' / (g + sigma2),  x_lmmse = w*y = mu*x + noise,  mu = g/(g + sigma2)
%   LMMSE shrinks the symbol by mu; the soft demapper needs it unbiased:
%       xhat = x_lmmse / mu = sum_r conj(H_r) Y_r / g
%       nu   = (1 - mu)/mu   = sigma2 / g
%   (nu large where the channel fades). For one stream this is exactly
%   maximum-ratio combining, which is how it is computed below.
%   Outputs keep only data REs, in transmit order: D x B, D = numel(ctx.DataIndices).
%   Products and sums only: runs on dlarray.
B = size(Y, 4);
gain = sum(real(H).^2 + imag(H).^2, 3);  % F x N x 1 x B
combined = sum(conj(H) .* Y, 3);
FN = ctx.NumSubcarriers * ctx.NumSymbols;
gain = reshape(gain, FN, B);
combined = reshape(combined, FN, B);
gain = max(gain(ctx.DataIndices, :), eps);
xhat = combined(ctx.DataIndices, :) ./ gain;
nu = reshape(sigma2, 1, B) ./ gain;
end
