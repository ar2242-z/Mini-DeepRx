function res = resources(carrier, cfg)
%RESOURCES Data and DMRS positions in the slot, plus the DMRS symbols.
%   res = tx.resources(carrier, cfg) uses a PUSCH with L = cfg.NumLayers
%   layers over all resource blocks and 14 symbols, mapping type A, with the
%   DMRS settings in cfg (TS 38.211 6.4.1.1). Fields, one column per layer:
%     DataIndices   D x L linear indices into the F x N x L layer grid; all
%                   layers use the same D resource elements
%     PilotIndices  P x L linear indices of each layer's DMRS port
%     Pilots        P x L DMRS symbols at PilotIndices
%   With one layer these are the columns they always were. The DMRS ports
%   are the toolbox default 0..L-1.
% Unused CDM groups stay empty. Their 3 dB DMRS power boost is not applied:
% pilots keep unit magnitude, which the LS receiver relies on.
pusch = nrPUSCHConfig;
pusch.PRBSet = 0:carrier.NSizeGrid-1; % Toolbox PRB numbers start at zero.
pusch.SymbolAllocation = [0 14];    % Start at symbol 0; use 14 symbols.
pusch.MappingType = 'A';
pusch.NumLayers = cfg.NumLayers;
pusch.Modulation = cfg.Modulation;
pusch.TransformPrecoding = false;
pusch.EnablePTRS = false;
pusch.DMRS.DMRSTypeAPosition = cfg.DMRSTypeAPosition; % 2 means MATLAB column 3.
pusch.DMRS.DMRSLength = cfg.DMRSLength;
pusch.DMRS.DMRSAdditionalPosition = cfg.DMRSAdditionalPosition; % Extra pilot symbols.
pusch.DMRS.DMRSConfigurationType = cfg.DMRSConfigurationType;
pusch.DMRS.NumCDMGroupsWithoutData = cfg.NumCDMGroupsWithoutData;
res.DataIndices = nrPUSCHIndices(carrier, pusch);
res.PilotIndices = nrPUSCHDMRSIndices(carrier, pusch);
res.Pilots = nrPUSCHDMRS(carrier, pusch);
end
