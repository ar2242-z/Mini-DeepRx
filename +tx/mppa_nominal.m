function pa = mppa_nominal()
%MPPA_NOMINAL The nominal 3GPP memory polynomial PA, normalized (MPPA.md).
%   pa = tx.mppa_nominal() returns the 28 nm CMOS PA at 31 GHz (KU Leuven)
%   of 3GPP TR 38.803 Annex A.1 B), "PA model for above 6 GHz":
%       y(n) = sum_{k=1..K} sum_{m=0..M-1} a_km x(n-m)|x(n-m)|^(k-1)
%   with K = 8 and M = 5. The TR lists the 40 a_km as one vector, k-major
%   (the 5 taps of k = 1, then of k = 2, ...): this order gives a sensible
%   linear filter for k = 1, the other a small-signal gain of 1.8e5.
%   The TR states no sample rate; its linear taps are flat only in the
%   middle quarter of the band, so tx.mppa runs it oversampled.
%   Normalized here so that the static AM/AM curve (a constant input
%   through all taps) first peaks at input 1 with output 1, and the
%   small-signal gain has zero phase:
%       v(n) = sum_k sum_m B(k,m+1) u(n-m)|u(n-m)|^(k-1),  |u| <= 1
%   pa fields:
%     Raw          K x M a_km of the TR
%     B            K x M normalized coefficients
%     InputScale   |x| of the TR model at saturation (u = x/InputScale)
%     OutputScale  |y| of the TR model at saturation
%     Basis        K x K upper-triangular R: W = R*B are the coefficients
%                  in polynomials orthonormal on 0 <= |u| <= 1 (Gram-Schmidt
%                  of u, u^2, ..., u^K), where nn.pa_bank dithers them
%     K, M
persistent cached
if isempty(cached)
    cached = build();
end
pa = cached;
end

function pa = build()
    a = [-10.0624+14.6485i, 24.6983-25.5192i, -28.6702+24.4684i, 18.9709-12.0500i, ...
        -5.3080+2.4235i, -63.1123+9.4912i, -18.2854-9.0971i, -33.6220-0.1089i, ...
        28.2194-25.5253i, -16.7754+26.7834i, 1.2797e+03-1.9632e+02i, 5.6546e+02+1.4583e+02i, ...
        6.7368e+02-6.4518e+01i, -1.0422e+03+7.0243e+02i, 5.1510e+02-6.8161e+02i, ...
        -1.1576e+04+9.9815e+02i, -8.6173e+03-6.5709e+02i, -5.8579e+03+1.7786e+02i, ...
        1.4834e+04-8.9848e+03i, -7.2260e+03+8.4931e+03i, 5.7725e+04-1.6098e+03i, ...
        6.1271e+04+2.1093e+03i, 3.0381e+04+1.1418e+03i, -1.1167e+05+6.4821e+04i, ...
        5.6209e+04-5.9278e+04i, -1.5194e+05-5.1975e+03i, -2.3711e+05-1.8474e+03i, ...
        -8.7976e+04-1.4443e+04i, 4.5408e+05-2.5881e+05i, -2.4165e+05+2.3195e+05i, ...
        1.8482e+05+2.2474e+04i, 4.7763e+05-9.1215e+03i, 1.3200e+05+5.0018e+04i, ...
        -9.4830e+05+5.3597e+05i, 5.3878e+05-4.7644e+05i, -6.4854e+04-2.2706e+04i, ...
        -3.9248e+05+1.9477e+04i, -7.9196e+04-5.8301e+04i, 7.9804e+05-4.4917e+05i, ...
        -4.8561e+05+4.0015e+05i];
    K = 8; M = 5;
    raw = reshape(a, M, K).';                 % raw(k, m+1) = a_km
    % Static curve of the TR model: first AM/AM peak (saturation).
    curve = tx.mppa_curve(raw, linspace(0, 1, 100001));
    r = curve.SaturationInput; y = curve.SaturationOutput;
    B = raw .* (r.^(1:K)).' / y;
    B = B * exp(-1i*angle(sum(B(1, :))));    % Zero-phase small-signal gain.
    u = linspace(0, 1, 2001).';
    [~, R] = qr(u.^(1:K)/sqrt(numel(u)), 0);
    pa = struct('Raw', raw, 'B', B, 'InputScale', r, 'OutputScale', y, ...
        'Basis', R, 'K', K, 'M', M);
end
