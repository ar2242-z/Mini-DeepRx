function [y, gain] = pa(x, cfg)
%PA Memoryless transmit power amplifier on the time waveform.
%   [y, gain] = tx.pa(x, cfg) uses cfg.PAModel, PABackoffdB, RappSmoothness
%   and PACoefficients. The input is scaled so its RMS sits
% PABackoffdB below the saturation amplitude (1). The output is scaled back
% to the input RMS, so SNRdB keeps its meaning and only distortion is added.
gain = 1;
if strcmp(cfg.PAModel,'none')
    y = x;
    return
end
rmsIn = sqrt(mean(abs(x).^2));
u = x/rmsIn * 10^(-cfg.PABackoffdB/20);
switch cfg.PAModel
    case 'rapp'
        p = cfg.RappSmoothness;
        v = u ./ (1 + abs(u).^(2*p)).^(1/(2*p));
    case 'polynomial'
        % Hold |u| at saturation so a falling AM/AM curve cannot fold back.
        u = min(abs(u),1) .* exp(1i*angle(u));
        a = cfg.PACoefficients;
        v = zeros(size(u));
        for k = 1:numel(a)
            v = v + a(k)*u.*abs(u).^(2*k-2);
        end
end
y = v * (rmsIn/sqrt(mean(abs(v).^2)));
% Bussgang gain: y = gain*x + distortion uncorrelated with x.
gain = (x'*y)/(x'*x);
end
