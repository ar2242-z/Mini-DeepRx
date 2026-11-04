function cfg = use_pa(cfg, coefficients)
%USE_PA Put one PA of nn.pa_bank into a link configuration.
%   cfg = nn.use_pa(cfg, coefficients): a bank row is PACoefficients for the
%   polynomial miniPA and MPCoefficients for the memory polynomial 'mp3gpp'.
if strcmp(cfg.PAModel, 'mp3gpp')
    cfg.MPCoefficients = coefficients;
else
    cfg.PACoefficients = coefficients;
end
end
