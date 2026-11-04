function bank = pa_bank(baseCfg, ncfg)
%PA_BANK Training and validation sets of slightly different PAs.
%   bank = nn.pa_bank(baseCfg, ncfg)
%   Physical meaning: real amplifiers of the same type differ a little, so
%   a receiver should learn to undo a family of PAs, not one curve.
%   Each PA adds complex Gaussian dither, ncfg.PADither x |a_k|, to every
%   polynomial coefficient a_k of baseCfg.PACoefficients (complex dither
%   also gives AM/PM). A draw is kept only if its AM/AM curve rises and
%   bends down up to saturation |u| = 1, like a real PA.
%   bank.Training: NumTrainingPAs x K, bank.Validation: NumValidationPAs x K,
%   drawn from one stream seeded by ncfg.PASeed, so the two never overlap.
%   Memory polynomial 'mp3gpp' (MPPA.md): the static curve of the Link's PA
%   (MPCoefficients, [] = tx.mppa_nominal), g_k = sum_m B(k,m), is dithered
%   in the orthonormal polynomial basis, w = R*g, w + PADither x |w| x noise,
%   and the change is added to the first tap B(:,1), so the 3GPP memory
%   terms stay as they are. Dithering the coefficients themselves fails:
%   powers of |u| cancel (each term up to ~2000x the output) and so do the
%   5 taps (the linear taps are ~20x their sum), so even 0.5% noise there
%   wrecks the curve. A draw is kept if its static
%   AM/AM rises up to its first peak (saturation) between |u| = 0.8 and
%   1.25, compresses by 1 dB before it, and bends upwards (gain expansion)
%   at most twice as much as the nominal PA. Rows are B(:).'.
%   Empty (no bank: every slot uses the Link's PA) unless the Link PA is
%   'polynomial' or 'mp3gpp' and PADither > 0.
bank = struct('Training', [], 'Validation', []);
if ~any(strcmp(baseCfg.PAModel, {'polynomial','mp3gpp'})) || ncfg.PADither == 0
    return
end
stream = RandStream('mrg32k3a', 'Seed', ncfg.PASeed);
if strcmp(baseCfg.PAModel, 'mp3gpp')
    nominal = tx.mppa_nominal();
    base = nominal.B;
    if ~isempty(baseCfg.MPCoefficients)
        base = reshape(baseCfg.MPCoefficients, nominal.K, nominal.M);
    end
    bank.Training = drawMP(base, nominal.Basis, ncfg.PADither, ncfg.NumTrainingPAs, stream);
    bank.Validation = drawMP(base, nominal.Basis, ncfg.PADither, ncfg.NumValidationPAs, stream);
    return
end
base = baseCfg.PACoefficients;
bank.Training = draw(base, ncfg.PADither, ncfg.NumTrainingPAs, stream);
bank.Validation = draw(base, ncfg.PADither, ncfg.NumValidationPAs, stream);
end

function pas = draw(base, dither, count, stream)
    u = linspace(0, 1, 201).';
    powers = u.^(2*(1:numel(base)) - 1);   % u, u^3, u^5, ...
    pas = complex(zeros(count, numel(base)));
    k = 0; attempts = 0;
    while k < count
        attempts = attempts + 1;
        assert(attempts <= 1000*count, 'Could not draw valid PAs; reduce PADither.');
        noise = (randn(stream, size(base)) + 1i*randn(stream, size(base)))/sqrt(2);
        a = base + dither*abs(base).*noise;
        amam = abs(powers*a(:));
        if all(diff(amam) > 0) && all(diff(amam, 2) < 1e-12)
            k = k + 1;
            pas(k, :) = a;
        end
    end
end

function pas = drawMP(base, R, dither, count, stream)
    u = (0:1e-3:1.5).';
    bend = @(c) diff(c.AMAM(c.Input <= c.SaturationInput), 2);
    limit = 2*max(bend(tx.mppa_curve(base, u)));
    w = R*sum(base, 2);
    pas = complex(zeros(count, numel(base)));
    k = 0; attempts = 0;
    while k < count
        attempts = attempts + 1;
        assert(attempts <= 1000*count, 'Could not draw valid PAs; reduce PADither.');
        noise = (randn(stream, size(w)) + 1i*randn(stream, size(w)))/sqrt(2);
        B = base;
        B(:, 1) = B(:, 1) + R \ (dither*abs(w).*noise);
        c = tx.mppa_curve(B, u);
        if c.SaturationInput >= 0.8 && c.SaturationInput <= 1.25 && ...
                ~isnan(c.P1dBOutput) && max(bend(c)) <= limit
            k = k + 1;
            pas(k, :) = B(:).';
        end
    end
end
