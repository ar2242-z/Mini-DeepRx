function cfg = override(cfg, settings)
%OVERRIDE Copy the fields of settings onto a link configuration and validate it.
%   cfg = sim.override(cfg, settings), e.g. a per-slot pilot layout:
%     cfg = sim.override(link_config(), struct('DMRSAdditionalPosition', 1));
%   link_config rejects unknown or inconsistent settings. Fields of cfg that
%   link_config no longer has (from configurations saved by older code, such
%   as a model's LinkConfig) are dropped.
known = fieldnames(link_config());
cfg = rmfield(cfg, setdiff(fieldnames(cfg), known));
for name = fieldnames(settings).'
    cfg.(name{1}) = settings.(name{1});
end
options = [fieldnames(cfg).'; struct2cell(cfg).'];
cfg = link_config(options{:});
end
