function [Hp, sigma2] = ls_estimate(Y, ctx)
%LS_ESTIMATE Least-squares channel estimate at the pilots, and noise power.
%   [Hp, sigma2] = rx.ls_estimate(Y, ctx), Y is F x N x Nr x B.
%   Hp: P x K x Nr x B x L channel at the P pilot subcarriers of the K
%       pilot symbols, from each layer (dimension 5; one layer: P x K x Nr
%       x B). With unit-power pilots, Y = H*X + n gives
%       Hp = Y .* conj(X) = H + noise (no division needed).
%   sigma2: 1 x 1 x 1 x B noise (plus interference) power per resource
%       element. Second differences d = h(k-1) - 2h(k) + h(k+1) over equally
%       spaced pilots cancel a locally linear channel, leaving noise with
%       E|d|^2 = (1+4+1)*sigma^2. Averaged over pilots, pilot symbols,
%       antennas and layers. Channel curvature and Doppler bias it
%       slightly upward.
%   One layer only for now: the layers' DMRS ports share their resource
%   elements (code division), so Y .* conj(X) is a mix of the layers'
%   channels until the ports are separated (SU-MIMO stage 2). The shapes
%   are already those of L layers.
%   Uses only indexing and products, so it also runs on dlarray.
assert(ctx.NumLayers == 1, ...
    'rx.ls_estimate cannot separate the DMRS ports of %d layers yet.', ctx.NumLayers);
[~, ~, Nr, B] = size(Y);
L = ctx.NumLayers;
P = numel(ctx.PilotRows); K = numel(ctx.PilotCols);
Hp = Y(ctx.PilotRows, ctx.PilotCols, :, :) .* conj(ctx.PilotValues); % P x K x Nr x B x L
d = ctx.SecondDifference * reshape(Hp, P, K*Nr*B*L);
d = reshape(permute(reshape(d, [], K, Nr, B, L), [1 2 3 5 4]), [], B);
sigma2 = reshape(mean(real(d).^2 + imag(d).^2, 1)/6, 1, 1, 1, B);
end
