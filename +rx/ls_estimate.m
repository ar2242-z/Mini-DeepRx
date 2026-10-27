function [Hp, sigma2] = ls_estimate(Y, ctx)
%LS_ESTIMATE Least-squares channel estimate at the pilots, and noise power.
%   [Hp, sigma2] = rx.ls_estimate(Y, ctx), Y is F x N x Nr x B.
%   Hp: P x K x Nr x B channel at the P pilot subcarriers of the K pilot
%       symbols. With unit-power pilots, Y = H*X + n gives
%       Hp = Y .* conj(X) = H + noise (no division needed).
%   sigma2: 1 x 1 x 1 x B noise (plus interference) power per resource
%       element. Second differences d = h(k-1) - 2h(k) + h(k+1) over equally
%       spaced pilots cancel a locally linear channel, leaving noise with
%       E|d|^2 = (1+4+1)*sigma^2. Averaged over pilots, pilot symbols and
%       antennas. Channel curvature and Doppler bias it slightly upward.
%   Uses only indexing and products, so it also runs on dlarray.
[~, ~, Nr, B] = size(Y);
P = numel(ctx.PilotRows); K = numel(ctx.PilotCols);
Hp = Y(ctx.PilotRows, ctx.PilotCols, :, :) .* conj(ctx.PilotValues);
d = ctx.SecondDifference * reshape(Hp, P, K*Nr*B);
d = reshape(d, [], B);
sigma2 = reshape(mean(real(d).^2 + imag(d).^2, 1)/6, 1, 1, 1, B);
end
