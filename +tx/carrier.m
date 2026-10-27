function carrier = carrier(cfg, cellID)
%CARRIER 5G carrier for one slot: numerology, bandwidth, and cell ID.
%   carrier = tx.carrier(cfg) uses cfg.NumResourceBlocks and
%   cfg.SubcarrierSpacing, normal cyclic prefix, slot 0, and cell ID 0.
%   carrier = tx.carrier(cfg, cellID) sets another cell ID (DMRS scrambling).
if nargin < 2
    cellID = 0;
end
carrier = nrCarrierConfig;
carrier.NSizeGrid = cfg.NumResourceBlocks;
carrier.SubcarrierSpacing = cfg.SubcarrierSpacing;
carrier.CyclicPrefix = 'normal';
carrier.NSlot = 0;
carrier.NCellID = cellID;
end
