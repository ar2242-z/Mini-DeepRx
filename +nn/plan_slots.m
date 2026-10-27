function plan = plan_slots(ncfg, roots, slotsPerRoot, epochs, excluded, pas)
%PLAN_SLOTS Conditions and seeds for every slot of a dataset.
%   plan = nn.plan_slots(ncfg, roots, slotsPerRoot, epochs, excluded, pas)
%   For each master seed in roots and each epoch, draws slotsPerRoot slots.
%   Each slot gets a unique Seed (never in excluded, never repeated) and
%   conditions drawn uniformly from the ncfg ranges: SNR, Doppler, delay
%   spread, channel model, pilot layout, SIR and PA back-off.
%   pas (optional, from nn.pa_bank): one PA per row; each slot draws one.
%   plan: table with Epoch, MasterSeed, Seed, SNRdB, DopplerHz, DelaySpread,
%   ChannelModel, PilotIndex, SIRdB, PABackoffdB, PAIndex, PACoefficients
%   (PAIndex 0 and empty coefficients: the Link's own PA). Pass a
%   validation plan's seeds as excluded for training, so the sets never
%   share a slot.
if nargin < 6
    pas = [];
end
uniform = @(s, r) r(1) + diff(r)*rand(s);
rows = cell(numel(roots)*epochs*slotsPerRoot, 12);
used = excluded(:);
k = 0;
for root = roots(:).'
    seedStream = RandStream('mt19937ar', 'Seed', root);
    conditionStream = RandStream('mrg32k3a', 'Seed', root);
    % PAs from their own substream: plans without a bank are unchanged.
    paStream = RandStream('mrg32k3a', 'Seed', root);
    paStream.Substream = 2;
    for epoch = 1:epochs
        for slot = 1:slotsPerRoot
            seed = randi(seedStream, [0 2^32-1]);
            while ismember(seed, used), seed = randi(seedStream, [0 2^32-1]); end
            used(end+1, 1) = seed; %#ok<AGROW>
            model = ncfg.ChannelModels{randi(conditionStream, numel(ncfg.ChannelModels))};
            pilot = randi(conditionStream, numel(ncfg.PilotConfigs));
            if isequal(ncfg.SIRRange, [Inf Inf]), sir = Inf;
            else, sir = uniform(conditionStream, ncfg.SIRRange); end
            pa = 0; coefficients = [];
            if ~isempty(pas)
                pa = randi(paStream, size(pas, 1));
                coefficients = pas(pa, :);
            end
            k = k + 1;
            rows(k, :) = {epoch, root, seed, uniform(conditionStream, ncfg.SNRRange), ...
                uniform(conditionStream, ncfg.DopplerRange), ...
                uniform(conditionStream, ncfg.DelaySpreadRange), model, pilot, sir, ...
                uniform(conditionStream, ncfg.PABackoffRange), pa, {coefficients}};
        end
    end
end
plan = cell2table(rows, 'VariableNames', {'Epoch','MasterSeed','Seed','SNRdB', ...
    'DopplerHz','DelaySpread','ChannelModel','PilotIndex','SIRdB','PABackoffdB', ...
    'PAIndex','PACoefficients'});
end
