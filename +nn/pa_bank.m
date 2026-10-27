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
%   Empty (no bank: every slot uses the Link's PA) unless the Link PA is
%   'polynomial' and PADither > 0.
bank = struct('Training', [], 'Validation', []);
if ~strcmp(baseCfg.PAModel, 'polynomial') || ncfg.PADither == 0
    return
end
stream = RandStream('mrg32k3a', 'Seed', ncfg.PASeed);
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
