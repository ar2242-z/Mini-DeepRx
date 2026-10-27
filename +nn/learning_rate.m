function rate = learning_rate(ncfg, step, totalSteps)
%LEARNING_RATE Adam learning rate at one training step (see nn_config).
%   rate = nn.learning_rate(ncfg, step, totalSteps), step = 1..totalSteps.
%     warmup   steps 1..WarmupSteps: LearningRate*step/WarmupSteps
%     hold     LearningRate until the decay starts at
%              max(WarmupSteps, round(DecayStart*totalSteps))
%     decay    'linear' or 'cosine' from LearningRate to FinalLearningRate,
%              reached at step totalSteps; 'constant' has none
%   rate = nn.learning_rate(ncfg, step, totalSteps) with
%   ncfg.LearningRateSchedule = 'range' gives the range test's rate, rising
%   exponentially over ncfg.RangeTestRates from step 1 to totalSteps.
peak = ncfg.LearningRate;
if strcmp(ncfg.LearningRateSchedule, 'range')
    r = ncfg.RangeTestRates;
    rate = r(1)*(r(2)/r(1))^((step - 1)/max(totalSteps - 1, 1));
    return
end
warmup = ncfg.WarmupSteps;
if step <= warmup
    rate = peak*step/warmup;
    return
end
decayFrom = max(warmup, round(ncfg.DecayStart*totalSteps));
if strcmp(ncfg.LearningRateSchedule, 'constant') || step <= decayFrom || totalSteps <= decayFrom
    rate = peak;
    return
end
t = min((step - decayFrom)/(totalSteps - decayFrom), 1);
final = ncfg.FinalLearningRate;
switch ncfg.LearningRateSchedule
    case 'linear'
        rate = peak + (final - peak)*t;
    case 'cosine'
        rate = final + (peak - final)*(1 + cos(pi*t))/2;
end
end
