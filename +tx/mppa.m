function [y, gain, info] = mppa(x, carrier, cfg)
%MPPA 3GPP memory polynomial transmit PA on the time waveform (MPPA.md).
%   [y, gain, info] = tx.mppa(x, carrier, cfg) uses cfg.MPCoefficients
%   ([] = tx.mppa_nominal), PABackoffdB and PAOversampling.
%   1. Upsample x by L = PAOversampling: the TR's memory taps only make
%      sense where the signal fills the middle of the band.
%   2. Scale the input so the mean output power sits PABackoffdB below the
%      PA's output power at its 1 dB compression point (output back-off,
%      OBO). Input amplitudes are held at the PA's saturation input, so
%      the output stays flat past saturation.
%   3. Memory polynomial, then low-pass and decimate back by L.
%   4. Rescale to the input RMS, so SNRdB keeps its meaning.
%   gain: F x 1 Bussgang gain of each subcarrier, y = gain .* x on the grid
%   plus distortion uncorrelated with x (least-squares linear filter from x
%   to y). info: InputScale (drive), OBOdB (achieved), Curve (tx.mppa_curve),
%   Input and Output (the PA's own samples at L times the sample rate).
nominal = tx.mppa_nominal();
if isempty(cfg.MPCoefficients)
    B = nominal.B;
else
    B = reshape(cfg.MPCoefficients, nominal.K, nominal.M);
end
curve = tx.mppa_curve(B);
assert(~isnan(curve.P1dBOutput), 'This PA never reaches 1 dB compression.');
L = cfg.PAOversampling;
rmsIn = sqrt(mean(abs(x).^2));
xu = resample(x, L, 1);
xu = xu / sqrt(mean(abs(xu).^2));

% Output back-off: bisection on the drive in dB (output power rises with it).
target = curve.P1dBOutput^2 * 10^(-cfg.PABackoffdB/10);
power = @(driveDB) mean(abs(amplify(B, xu*10^(driveDB/20), curve.SaturationInput)).^2);
lo = -40; hi = 20;
assert(power(hi) >= target, ['Output back-off %.2f dB is out of reach: the PA ' ...
    'saturates %.2f dB above its P1dB.'], cfg.PABackoffdB, ...
    10*log10(power(hi)/curve.P1dBOutput^2));
for it = 1:40
    mid = (lo + hi)/2;
    if power(mid) < target, lo = mid; else, hi = mid; end
end
drive = 10^((lo + hi)/40);
vu = amplify(B, xu*drive, curve.SaturationInput);
info.InputScale = drive;
info.OBOdB = 10*log10(curve.P1dBOutput^2 / mean(abs(vu).^2));
info.Curve = curve;
info.Input = xu*drive;
info.Output = vu;

v = resample(vu, 1, L);
v = v(1:numel(x));
y = v * (rmsIn/sqrt(mean(abs(v).^2)));
gain = bussgang(x, y, carrier.NSizeGrid*12, cfg.Nfft);
end

function v = amplify(B, u, saturation)
    % v(n) = sum_k sum_m B(k,m+1) uc(n-m)|uc(n-m)|^(k-1), |uc| = min(|u|, saturation).
    a = min(abs(u), saturation);
    uc = a .* exp(1i*angle(u));
    v = zeros(size(u));
    term = uc;
    for k = 1:size(B, 1)
        v = v + filter(B(k, :), 1, term);
        term = term .* a;
    end
end

function G = bussgang(x, y, F, Nfft)
    % Least-squares FIR h, taps -T..T, with y ~ h*x; G = its response on the
    % F active subcarriers (subcarrier k at (k - 1 - F/2) * spacing).
    T = 8;
    n = numel(x);
    X = zeros(n, 2*T + 1);
    for d = -T:T                              % Column d holds x(i - d).
        if d >= 0
            X(:, d + T + 1) = [zeros(d, 1); x(1:n-d)];
        else
            X(:, d + T + 1) = [x(1-d:n); zeros(-d, 1)];
        end
    end
    h = X \ y;
    k = (0:F-1).' - F/2;
    G = exp(-2i*pi*k*(-T:T)/Nfft) * h;
end
