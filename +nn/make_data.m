function data = make_data(baseCfg, ncfg, plan, keepTruth)
%MAKE_DATA Simulate the slots of a plan and keep what training needs.
%   data = nn.make_data(baseCfg, ncfg, plan, keepTruth)
%   For each plan row: run_link (Tx + channel), then rx.sync with the
%   receiver's own timing (cfg.TimingEstimation). Storing the synced time
%   samples serves every receiver: FCNN applies the FFT, TCNN works before it.
%   data fields (S = number of slots):
%     Slots      SlotLength x Nr x S single, synced received samples
%     PilotGrid  F x N x 1 x S single, known pilots of each slot's layout
%     Labels     F x N x Q x S single coded-bit planes (nn.labels)
%     Mask       F x N x Q x S logical, data REs
%     PilotIndex S x 1, Contexts (rx.context per pilot layout), Plan
%     Truth      S x 1 struct (InfoBits, CodedBits, Code) when keepTruth,
%                for coded evaluation (BLER)
if nargin < 4
    keepTruth = false;
end
for k = 1:numel(ncfg.PilotConfigs)
    contexts(k, 1) = rx.context(slot_config(baseCfg, ncfg.PilotConfigs(k))); %#ok<AGROW>
end
c = contexts(1);
S = height(plan);
Q = c.BitsPerSymbol;
data.Slots = complex(zeros(c.SlotLength, c.NumReceiveAntennas, S, 'single'));
data.PilotGrid = complex(zeros(c.NumSubcarriers, c.NumSymbols, 1, S, 'single'));
data.Labels = zeros(c.NumSubcarriers, c.NumSymbols, Q, S, 'single');
data.Mask = false(c.NumSubcarriers, c.NumSymbols, Q, S);
data.PilotIndex = plan.PilotIndex;
data.Contexts = contexts;
data.Plan = plan;
truth = struct('InfoBits', cell(S, 1), 'CodedBits', [], 'Code', []);
for s = 1:S
    cfg = nn.slot_cfg(baseCfg, ncfg, plan(s, :));
    ctx = contexts(plan.PilotIndex(s));
    [rxWaveform, t] = run_link(cfg);
    data.Slots(:, :, s) = rx.sync(rxWaveform, ctx, t.TimingOffset);
    data.PilotGrid(:, :, 1, s) = ctx.PilotGrid;
    [data.Labels(:, :, :, s), data.Mask(:, :, :, s)] = nn.labels(t.CodedBits, ctx);
    if keepTruth
        truth(s).InfoBits = t.InfoBits;
        truth(s).CodedBits = t.CodedBits;
        truth(s).Code = t.Code;
    end
end
if keepTruth
    data.Truth = truth;
end
end
