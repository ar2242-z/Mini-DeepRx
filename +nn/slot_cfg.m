function cfg = slot_cfg(baseCfg, ncfg, row)
%SLOT_CFG Link configuration of one planned slot.
%   cfg = nn.slot_cfg(baseCfg, ncfg, row) applies the pilot layout and the
%   conditions of one nn.plan_slots row to the base link configuration.
cfg = slot_config(baseCfg, ncfg.PilotConfigs(row.PilotIndex));
cfg.Seed = row.Seed;
cfg.SNRdB = row.SNRdB;
cfg.MaximumDopplerShift = row.DopplerHz;
cfg.DelaySpread = row.DelaySpread;
cfg.ChannelModel = char(row.ChannelModel);
cfg.SIRdB = row.SIRdB;
cfg.PABackoffdB = row.PABackoffdB;
if row.PAIndex > 0
    cfg.PACoefficients = row.PACoefficients{1}; % This slot's PA from the bank.
end
end
