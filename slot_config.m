function cfg = slot_config(cfg,settings)
%SLOT_CONFIG Copy the fields of settings onto cfg and validate the result.
% Used for per-slot pilot configurations, e.g.
%   cfg = slot_config(simo_config(),struct('DMRSAdditionalPosition',1));
% simo_config rejects unknown or inconsistent fields.
    for name = fieldnames(settings).'
        cfg.(name{1}) = settings.(name{1});
    end
    options = [fieldnames(cfg).'; struct2cell(cfg).'];
    cfg = simo_config(options{:});
end
