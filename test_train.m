function passed = test_train()
%% Check the training options: schedule, clipping, Resume, range test
% Run: test_train       (CPU only, so it can run beside a GPU training)
% 1. nn.learning_rate: the default is the old constant rate; warmup ramps
%    from 0; 'linear' and 'cosine' hold the peak until DecayStart and reach
%    FinalLearningRate at the last step; the range test rises
%    exponentially; experiments/pa_schedule parses.
% 2. nn.clip_gradients caps each network's norm separately and keeps the
%    direction; a network under the cap is unchanged.
% 3. nn.train stopped after epoch 2 (an error, as from Ctrl+C) and resumed
%    from its onEpoch state ends exactly as an uninterrupted run with a
%    decaying rate; history logs the scheduled rate and gradient norms.
% 4. train_receiver: 2 epochs then 'Resume',true to 3 equals 3 epochs in
%    one go; Resume refuses changed settings.
% 5. train_receiver 'FindLearningRate',true: rising rates, a suggestion
%    below the lowest-loss rate, saved to models/<Name>_lr_range.mat.

passed = true;
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'experiments'));

%% 1. Learning-rate schedule
base = nn_config('LearningRate', 2e-3);
constant = arrayfun(@(i) nn.learning_rate(base, i, 100), 1:100);
w = nn_config('LearningRate', 2e-3, 'WarmupSteps', 10, 'LearningRateSchedule', 'linear', ...
    'DecayStart', 0.3, 'FinalLearningRate', 1e-4);
r = arrayfun(@(i) nn.learning_rate(w, i, 100), 1:100);
c = w; c.LearningRateSchedule = 'cosine';
rc = arrayfun(@(i) nn.learning_rate(c, i, 100), 1:100);
g = w; g.LearningRateSchedule = 'range'; g.RangeTestRates = [1e-5 1e-1];
rr = arrayfun(@(i) nn.learning_rate(g, i, 50), 1:50);
near = @(a, b) abs(a - b) <= 1e-12 + 1e-9*abs(b);
ok = all(constant == 2e-3) && near(r(1), 2e-4) && near(r(10), 2e-3) && all(diff(r(1:10)) > 0) && ...
    all(r(10:30) == 2e-3) && near(r(65), (2e-3 + 1e-4)/2) && near(r(100), 1e-4) && ...
    all(diff(r(30:100)) < 0) && near(rc(65), (2e-3 + 1e-4)/2) && near(rc(100), 1e-4) && ...
    near(rr(1), 1e-5) && near(rr(50), 1e-1) && all(diff(log(rr)) > 0);
one = nn_config('LearningRate', 1e-3, 'WarmupSteps', 5, 'LearningRateSchedule', 'linear', ...
    'DecayStart', 0, 'FinalLearningRate', 0);
ok = ok && nn.learning_rate(one, 5, 20) == 1e-3 && near(nn.learning_rate(one, 20, 20), 0);
args = pa_schedule(); s = nn_config(args{:});
ok = ok && s.MiniBatchSize == 10 && strcmp(s.LearningRateSchedule, 'linear') && ...
    ~isempty(s.Link) && isequal(s.PABackoffRange, [0 10]);
try
    nn_config('LearningRate', 1e-3, 'FinalLearningRate', 2e-3); ok = false;
catch
end
passed = report(ok, sprintf(['1. schedule: constant default; warmup 0.2 -> 2 (x1e-3) over 10 steps; ' ...
    'hold to 30%%; linear/cosine mid %.3g/%.3g, end %.0e; range 1e-5 -> 1e-1'], r(65), rc(65), r(100))) && passed;

%% 2. Gradient clipping per network
grads.a = table({'l1'; 'l2'}, {'Weights'; 'Bias'}, {dlarray([3 0]); dlarray(4)}, ...
    'VariableNames', {'Layer', 'Parameter', 'Value'});
grads.b = table({'l1'}, {'Weights'}, {dlarray([0.3 0.4])}, ...
    'VariableNames', {'Layer', 'Parameter', 'Value'});
[clipped, norms] = nn.clip_gradients(grads, 1);
[same, measured] = nn.clip_gradients(grads, Inf);
va = extractdata(clipped.a.Value{1}); vb = extractdata(clipped.a.Value{2});
ok = near(norms.a, 5) && near(norms.b, 0.5) && near(norm([va vb]), 1) && near(va(1), 0.6) && ...
    isequal(extractdata(clipped.b.Value{1}), [0.3 0.4]) && ...
    isequal(extractdata(same.a.Value{1}), [3 0]) && near(measured.a, 5);
passed = report(ok, '2. clipping: norm 5 -> 1 (direction kept), norm 0.5 untouched, Inf only measures') && passed;

%% 3. nn.train: interrupted and resumed = uninterrupted
quiet = {'ShowPlots',false,'RunChecks',false,'SaveData',false,'Verbose',false};
ncfg = nn_config('Link', {'Coding','ldpc'}, 'SIRRange', [Inf Inf], 'MaxEpochs', 4, ...
    'MiniBatchSize', 2, 'LearningRate', 3e-3, 'LearningRateSchedule', 'linear', ...
    'WarmupSteps', 3, 'FinalLearningRate', 1e-4, 'GradientClip', 0.5, ...
    'Widths', [8 8], 'Dilations', [1 1; 1 1], 'ExecutionEnvironment', 'cpu', 'Verbose', false);
baseCfg = simo_config(quiet{:}, ncfg.Link{:});
ctx = rx.context(baseCfg);
plan = nn.plan_slots(ncfg, 77, 3, ncfg.MaxEpochs, []);
for e = 1:ncfg.MaxEpochs
    epochs{e} = nn.make_data(baseCfg, ncfg, plan(plan.Epoch == e, :)); %#ok<AGROW>
end
validation = nn.make_data(baseCfg, ncfg, nn.plan_slots(ncfg, 99, 2, 1, plan.Seed), true);
inputSize = [ctx.NumSubcarriers ctx.NumSymbols 2*(2*baseCfg.NumReceiveAntennas + 1)];
rng(7, 'twister');
params = struct('fcnn', nn.fcnn(inputSize, ctx.BitsPerSymbol, ncfg.Widths, ncfg.Dilations));
forward = @(p, b, t) nn.fcnn_forward(p, b, t, ctx);
rng(11, 'twister');
[whole, fullHistory] = nn.train(params, forward, @(e) epochs{e}, validation, ncfg);
store = containers.Map();
rng(11, 'twister');
try
    nn.train(params, forward, @(e) interruptAt(epochs, e, 3), validation, ncfg, [], [], ...
        @(state) keep(store, state));
    interrupted = false;
catch
    interrupted = true;
end
state = store('state');
rng(12345, 'twister');   % Resume must restore the random state itself.
[resumed, resumedHistory] = nn.train(params, forward, @(e) epochs{e}, validation, ncfg, ...
    [], [], [], state);
gap = max(cellfun(@(a, b) max(abs(extractdata(a) - extractdata(b)), [], 'all'), ...
    whole.Params.fcnn.Learnables.Value, resumed.Params.fcnn.Learnables.Value));
steps = ceil(3/ncfg.MiniBatchSize)*(1:4);
expected = arrayfun(@(i) nn.learning_rate(ncfg, i, steps(end)), steps).';
ok = interrupted && state.Epoch == 2 && gap == 0 && whole.Epoch == resumed.Epoch && ...
    isequal(fullHistory{:, 1:6}, resumedHistory{:, 1:6}) && ...
    max(abs(fullHistory.LearningRate - expected)) < 1e-15 && all(fullHistory.GradientNorm > 0);
passed = report(ok, sprintf(['3. nn.train stopped at epoch 3, resumed from epoch 2: weights differ by %g, ' ...
    'history equal; logged rates %s'], gap, mat2str(fullHistory.LearningRate.', 3))) && passed;

%% 4. train_receiver Resume
names = {'test_train_full', 'test_train_part', 'test_train_range'};
files = [cellfun(@(n) fullfile(root, 'models', [n '.mat']), names, 'UniformOutput', false), ...
    cellfun(@(n) fullfile(root, 'models', [n '_last.mat']), names, 'UniformOutput', false), ...
    {fullfile(root, 'models', 'test_train_range_lr_range.mat')}];
cleanup = onCleanup(@() deleteIfExists(files));
small = {'Link', {'Coding','ldpc'}, 'SIRRange', [Inf Inf], 'SlotsPerSeed', 1, ...
    'ValidationSlotsPerSeed', 1, 'MiniBatchSize', 2, 'LearningRate', 3e-3, ...
    'WarmupSteps', 4, 'GradientClip', 1, 'Widths', [8 8], 'Dilations', [1 1; 1 1], ...
    'ExecutionEnvironment', 'cpu', 'Verbose', false};
evalc('[mFull, hFull] = train_receiver(small{:}, ''Flow'', ''fcnn'', ''Name'', ''test_train_full'', ''MaxEpochs'', 3);');
evalc('train_receiver(small{:}, ''Flow'', ''fcnn'', ''Name'', ''test_train_part'', ''MaxEpochs'', 2);');
evalc(['[mPart, hPart] = train_receiver(small{:}, ''Flow'', ''fcnn'', ''Name'', ''test_train_part'', ' ...
    '''MaxEpochs'', 3, ''Resume'', true);']);
gap = max(cellfun(@(a, b) max(abs(extractdata(a) - extractdata(b)), [], 'all'), ...
    mFull.Nets.fcnn.Learnables.Value, mPart.Nets.fcnn.Learnables.Value));
refused = false;
try
    evalc(['train_receiver(small{:}, ''Flow'', ''fcnn'', ''Name'', ''test_train_part'', ' ...
        '''MaxEpochs'', 4, ''Resume'', true, ''MiniBatchSize'', 3);']);
catch err
    refused = contains(err.message, 'MiniBatchSize');
end
ok = gap == 0 && height(hPart) == 3 && isequal(hFull{:, 1:6}, hPart{:, 1:6}) && ...
    mFull.BestEpoch == mPart.BestEpoch && refused;
passed = report(ok, sprintf(['4. train_receiver: 2 epochs + Resume to 3 = 3 epochs (weights differ by %g); ' ...
    'a changed MiniBatchSize is refused'], gap)) && passed;

%% 5. Learning-rate range test
evalc(['[result, steps] = train_receiver(small{:}, ''Flow'', ''fcnn'', ''Name'', ''test_train_range'', ' ...
    '''FindLearningRate'', true, ''RangeTestSteps'', 20, ''RangeTestRates'', [1e-5 1]);']);
ok = isfile(files{end}) && height(steps) == result.Steps && result.Steps <= 20 && ...
    all(diff(steps.LearningRate) > 0) && near(steps.LearningRate(1), 1e-5) && ...
    result.LearningRate == result.MinimumLossRate/10 && isfinite(result.LearningRate);
passed = report(ok, sprintf(['5. range test: %d steps, lowest loss at lr %.2g, stopped at %.2g ' ...
    '(diverged: %d); suggests %.2g'], result.Steps, result.MinimumLossRate, result.StoppedAt, ...
    result.Diverged, result.LearningRate)) && passed;

if passed, fprintf('All training-option checks passed.\n'); end
end

%% Local helpers
function data = interruptAt(epochs, e, stop)
    if e == stop, error('test:interrupt', 'Interrupted at epoch %d.', e); end
    data = epochs{e};
end

function keep(store, state)
    store('state') = state; %#ok<NASGU> (containers.Map is a handle)
end

function deleteIfExists(files)
    for k = 1:numel(files)
        if isfile(files{k}), delete(files{k}); end
    end
end

function ok = report(ok, label)
    if ok, fprintf('  PASS  %s\n', label); else, fprintf('  FAIL  %s\n', label); end
end
