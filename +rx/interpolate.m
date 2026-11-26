function H = interpolate(Hp, ctx)
%INTERPOLATE Spread the pilot channel estimates over the whole grid.
%   H = rx.interpolate(Hp, ctx), Hp is P x K x Nr x B x L from
%   rx.ls_estimate; H is F x N x Nr x B x L (L layers; one layer: no fifth
%   dimension). As the DMRS-based baseline of Pihlajasalo et al.
%   (IEEE TWC 2023, Sec. V): the LS estimates interpolated linearly over
%   the slot. Linear in frequency between pilot subcarriers (extrapolated
%   at the band edges), then linear in time between pilot symbols, with the
%   nearest pilot symbol held before the first and after the last:
%       H = FrequencyInterp * Hp * TimeInterp.'   per antenna, slot and layer.
%   With one pilot symbol the channel is assumed constant over the slot,
%   so Doppler is not tracked. Each layer uses its own pilot subcarriers
%   (ctx.FrequencyInterp(:, :, l)). Matrix products only: runs on dlarray.
[P, K, Nr, B, L] = size(Hp);
F = ctx.NumSubcarriers; N = ctx.NumSymbols;
M = Nr*B*L;                     % Antennas, slots and layers alike.
if L == 1
    Hf = ctx.FrequencyInterp(:, :, 1) * reshape(Hp, P, K*M);  % F x (K*M)
else
    Hf = cell(1, L);
    for l = 1:L
        Hf{l} = ctx.FrequencyInterp(:, :, l) * reshape(Hp(:, :, :, :, l), P, K*Nr*B);
    end
    Hf = cat(2, Hf{:});                                       % F x (K*M), layer slowest
end
Hf = reshape(permute(reshape(Hf, F, K, M), [2 1 3]), K, []); % K x (F*M)
H = ctx.TimeInterp * Hf;                                      % N x (F*M)
H = permute(reshape(H, N, F, Nr, B, L), [2 1 3 4 5]);
end
