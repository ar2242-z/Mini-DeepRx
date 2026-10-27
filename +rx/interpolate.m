function H = interpolate(Hp, ctx)
%INTERPOLATE Spread the pilot channel estimates over the whole grid.
%   H = rx.interpolate(Hp, ctx), Hp is P x K x Nr x B from rx.ls_estimate;
%   H is F x N x Nr x B. As the DMRS-based baseline of Pihlajasalo et al.
%   (IEEE TWC 2023, Sec. V): the LS estimates interpolated linearly over
%   the slot. Linear in frequency between pilot subcarriers (extrapolated
%   at the band edges), then linear in time between pilot symbols, with the
%   nearest pilot symbol held before the first and after the last:
%       H = FrequencyInterp * Hp * TimeInterp.'   per antenna and slot.
%   With one pilot symbol the channel is assumed constant over the slot,
%   so Doppler is not tracked. Matrix products only: runs on dlarray.
[P, K, Nr, B] = size(Hp);
F = ctx.NumSubcarriers; N = ctx.NumSymbols;
Hf = ctx.FrequencyInterp * reshape(Hp, P, K*Nr*B);              % F x (K*Nr*B)
Hf = reshape(permute(reshape(Hf, F, K, Nr*B), [2 1 3]), K, []); % K x (F*Nr*B)
H = ctx.TimeInterp * Hf;                                         % N x (F*Nr*B)
H = permute(reshape(H, N, F, Nr, B), [2 1 3 4]);
end
