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
%   Several layers: each layer is estimated at its own pilot REs
%   (ctx.PilotRows(:, l)). This separates the layers only when they sit on
%   different subcarriers (DMRS ports in different CDM groups, e.g. ports
%   0 and 2, with NumCDMGroupsWithoutData 2 so the other layer's comb
%   carries nothing). Ports of one CDM group share their REs and are told
%   apart by a cover code, so Y .* conj(X) would mix their channels: that
%   is refused (ctx.PilotsSeparable false).
%   Uses only indexing and products, so it also runs on dlarray.
assert(ctx.PilotsSeparable, ['rx.ls_estimate cannot separate DMRS ports that share ' ...
    'their resource elements (one CDM group); put each layer in its own CDM group, ' ...
    'e.g. DMRSPortSet [0 2] with NumCDMGroupsWithoutData 2.']);
[~, ~, Nr, B] = size(Y);
L = ctx.NumLayers;
P = size(ctx.PilotRows, 1); K = numel(ctx.PilotCols);
if L == 1
    Hp = Y(ctx.PilotRows, ctx.PilotCols, :, :) .* conj(ctx.PilotValues); % P x K x Nr x B
else
    Hp = cell(1, 1, 1, 1, L);
    for l = 1:L
        Hp{l} = Y(ctx.PilotRows(:, l), ctx.PilotCols, :, :) .* conj(ctx.PilotValues(:, :, 1, 1, l));
    end
    Hp = cat(5, Hp{:});                                                  % P x K x Nr x B x L
end
d = ctx.SecondDifference * reshape(Hp, P, K*Nr*B*L);
d = reshape(permute(reshape(d, [], K, Nr, B, L), [1 2 3 5 4]), [], B);
sigma2 = reshape(mean(real(d).^2 + imag(d).^2, 1)/6, 1, 1, 1, B);
end
