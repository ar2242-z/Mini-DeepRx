function [H, dof] = delay_estimate(Hp, ctx, window, ridge)
%DELAY_ESTIMATE Smoothed channel estimate: LS pilots fitted by a few delay taps.
%   [H, dof] = rx.delay_estimate(Hp, ctx)
%   [H, dof] = rx.delay_estimate(Hp, ctx, window, ridge)
%   Hp: P x K x Nr x B x L from rx.ls_estimate; H: F x N x Nr x B x L.
%   The channel is a sum of paths with delays inside `window` = [first last]
%   in seconds relative to the synchronized (strongest) path, on the
%   receiver's sample grid:
%       H(k) = sum_tau h(tau) exp(-2i*pi*k*tau/Nfft)
%   The taps h are the ridge least-squares fit to the LS estimates at the
%   pilots, and H is their response on every subcarrier. Fewer free
%   parameters than pilots, so most of the noise in Hp is averaged out, and
%   so is the PA distortion on the pilots, which per-pilot LS estimation
%   (rx.interpolate) absorbs into the channel. rx.dpoi needs that: with
%   Hp held exactly, the equalized pilots equal the transmitted ones and
%   show no distortion to fit.
%   window: default [-0.4e-6 2e-6] s, wide enough for a 300 ns RMS delay
%   spread (TDL-C) and a few samples of timing error. A wider window fits
%   more of a long channel but averages out less noise.
%   ridge: default 1e-3, relative to the number of pilots; the occupied
%   band is narrower than the sample rate, so sample-spaced taps are not
%   all identifiable without it.
%   Time: as rx.interpolate (linear between pilot symbols, nearest held).
%   dof: effective number of fitted parameters per pilot symbol and antenna
%   (the trace of the pilot-to-pilot smoother), for noise estimates made
%   from residuals at the pilots.
if nargin < 3 || isempty(window), window = [-0.4e-6 2e-6]; end
if nargin < 4 || isempty(ridge), ridge = 1e-3; end
F = ctx.NumSubcarriers;
P = numel(ctx.PilotRows);
fs = ctx.Nfft * ctx.Carrier.SubcarrierSpacing * 1e3;
k = (0:F-1).' - F/2;
taus = floor(window(1)*fs):ceil(window(2)*fs);
Ef = exp(-2i*pi*k*taus/ctx.Nfft);          % F x T
E = Ef(ctx.PilotRows, :);                   % P x T
S = Ef*(((E'*E) + ridge*P*eye(numel(taus))) \ E');   % F x P smoother
dof = real(trace(S(ctx.PilotRows, :)));
ctx.FrequencyInterp = S;
H = rx.interpolate(Hp, ctx);
end
