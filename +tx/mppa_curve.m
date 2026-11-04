function c = mppa_curve(B, u)
%MPPA_CURVE Static AM/AM and AM/PM curves of a memory polynomial PA.
%   c = tx.mppa_curve(B) for K x M coefficients B (tx.mppa_nominal) on
%   input amplitudes u (default 0:1e-4:1.5). Static: a constant input
%   passes all M taps, so the PA acts as the polynomial
%       v(u) = sum_k g_k u^k,  g_k = sum_m B(k,m).
%   c fields: Input (u), AMAM |v|, AMPM angle(v) - angle(g_1) (rad),
%   SmallSignalGain |g_1|; SaturationInput/Output at the first AM/AM peak
%   (the last u if the curve keeps rising); P1dBInput/Output where the gain
%   |v|/u first falls 1 dB below |g_1| before saturation (NaN if never).
if nargin < 2
    u = 0:1e-4:1.5;
end
u = u(:);
g = sum(B, 2);
v = (u.^(1:numel(g))) * g;
c.Input = u;
c.AMAM = abs(v);
c.AMPM = unwrap(angle(v)) - angle(g(1));
c.AMPM(1) = 0;
c.SmallSignalGain = abs(g(1));
peak = find(diff(c.AMAM) <= 0, 1);
if isempty(peak), peak = numel(u); end
c.SaturationInput = u(peak);
c.SaturationOutput = c.AMAM(peak);
gainDrop = c.AMAM(2:peak) ./ u(2:peak) < c.SmallSignalGain * 10^(-1/20);
i1 = find(gainDrop, 1) + 1;
if isempty(i1)
    c.P1dBInput = NaN; c.P1dBOutput = NaN;
else
    c.P1dBInput = u(i1); c.P1dBOutput = c.AMAM(i1);
end
end
