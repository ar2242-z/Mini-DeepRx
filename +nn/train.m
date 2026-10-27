function [best, history] = train(params, forwardFcn, trainingData, validation, ncfg, trainable, onBest, onEpoch, start)
%TRAIN Generic training loop for neural receivers (FCNN, TCNN, hybrids).
%   [best, history] = nn.train(params, forwardFcn, trainingData, validation, ncfg)
%   params      struct of dlnetworks, e.g. params.fcnn (and later params.tcnn)
%   forwardFcn  [logits, states] = forwardFcn(params, batch, training),
%               logits F x N x Q x B (logit of P(bit = 1)); states.(name) is
%               the new batch-norm state of each network run in training
%               mode. training is the trainable list while training (frozen
%               networks then run in prediction mode) and false for
%               validation (see nn.in_training).
%   trainingData  @(epoch) -> data from nn.make_data (fresh slots each epoch)
%   validation  fixed data from nn.make_data (with Truth for BLER)
%   ncfg        nn_config: MaxEpochs, MiniBatchSize, LearningRate and its
%               schedule (nn.learning_rate), GradientClip, ...
%   trainable   names of the networks to update (default: all fields of
%               params); the others are frozen.
%   onBest      optional @(best, history): called whenever validation loss
%               improves, e.g. to save a checkpoint, so an interrupted run
%               keeps its best model.
%   onEpoch     optional @(state): called after every epoch with everything
%               needed to continue (CPU arrays): Params, AverageGrad,
%               AverageSqGrad, Iteration, Epoch, History, Best, BestLoss,
%               RandomState.
%   start       optional state from onEpoch: continue that run at the next
%               epoch, as if it had never stopped. ncfg.MaxEpochs may be
%               larger than in the first call (the decay then stretches).
%   Loss: masked binary cross-entropy (nn.bce_loss) on the coded bits, Adam.
%   Keeps the parameters with the lowest validation loss (best).
%   history: Epoch, TrainingLoss, ValidationLoss, ValidationBER,
%   ValidationBLER (NaN without Truth), LearningRate (at the epoch's last
%   step), GradientNorm (median before clipping, one column per trained
%   network in trainable order), Seconds.
%   With ncfg.FindLearningRate it runs the learning-rate range test instead
%   (no validation): best.LearningRate is the suggested peak rate and
%   history has Step, LearningRate, Loss, SmoothedLoss, GradientNorm per step.
if nargin < 6 || isempty(trainable)
    trainable = fieldnames(params).';
end
if nargin < 7, onBest = []; end
if nargin < 8, onEpoch = []; end
if nargin < 9, start = []; end
useGPU = ~strcmp(ncfg.ExecutionEnvironment, 'cpu') && canUseGPU;
assert(~strcmp(ncfg.ExecutionEnvironment, 'gpu') || useGPU, 'GPU requested but unavailable.');
if ncfg.FindLearningRate
    [best, history] = rangeTest(params, forwardFcn, trainingData, ncfg, trainable, useGPU);
    return
end
names = {'Epoch','TrainingLoss','ValidationLoss','ValidationBER','ValidationBLER', ...
    'LearningRate','GradientNorm','Seconds'};
history = table('Size', [ncfg.MaxEpochs numel(names)], ...
    'VariableTypes', repmat({'double'}, 1, numel(names)), 'VariableNames', names);
history.GradientNorm = NaN(ncfg.MaxEpochs, numel(trainable));  % One column per network.
if isempty(start)
    opt = struct();
    for name = trainable
        opt.avg.(name{1}) = []; opt.avgSq.(name{1}) = [];
    end
    bestLoss = Inf; iteration = 0; first = 1; best = struct();
else
    assert(start.Epoch < ncfg.MaxEpochs, ...
        'The saved run already finished %d epochs; raise MaxEpochs to continue.', start.Epoch);
    params = start.Params;
    opt.avg = start.AverageGrad; opt.avgSq = start.AverageSqGrad;
    history(1:start.Epoch, :) = start.History(1:start.Epoch, names);
    bestLoss = start.BestLoss; best = start.Best;
    iteration = start.Iteration; first = start.Epoch + 1;
    rng(start.RandomState);
    if ncfg.Verbose
        fprintf('Resuming at epoch %d (step %d, best epoch %d).\n', first, iteration, best.Epoch);
    end
end
if useGPU
    params = moveNetworks(params, @gpuArray);
    opt.avg = moveOptimizer(opt.avg, @gpuArray); opt.avgSq = moveOptimizer(opt.avgSq, @gpuArray);
end
totalSteps = [];
for epoch = first:ncfg.MaxEpochs
    clock = tic;
    data = trainingData(epoch);
    S = size(data.Slots, 3);
    if isempty(totalSteps), totalSteps = ncfg.MaxEpochs*ceil(S/ncfg.MiniBatchSize); end
    order = randperm(S);
    lossSum = 0; bitSum = 0; norms = zeros(0, numel(trainable));
    for firstSlot = 1:ncfg.MiniBatchSize:S
        idx = order(firstSlot:min(firstSlot + ncfg.MiniBatchSize - 1, S));
        iteration = iteration + 1;
        rate = nn.learning_rate(ncfg, iteration, totalSteps);
        [params, opt, value, norms(end+1, :)] = update(params, opt, forwardFcn, ...
            makeBatch(data, idx, useGPU), trainable, iteration, rate, ncfg.GradientClip); %#ok<AGROW>
        assert(isfinite(value), 'Non-finite training loss at iteration %d.', iteration);
        bits = nnz(data.Mask(:, :, :, idx));
        lossSum = lossSum + value*bits; bitSum = bitSum + bits;
    end
    [vLoss, vBER, vBLER] = validate(params, forwardFcn, validation, ncfg.MiniBatchSize, useGPU);
    history(epoch, :) = {epoch, lossSum/bitSum, vLoss, vBER, vBLER, rate, median(norms, 1), toc(clock)};
    if vLoss < bestLoss
        bestLoss = vLoss;
        best.Params = moveNetworks(params, @gather); % CPU-portable.
        best.Epoch = epoch;
        if ~isempty(onBest)
            onBest(best, history(1:epoch, :));
        end
    end
    if ~isempty(onEpoch)
        onEpoch(struct('Params', moveNetworks(params, @gather), ...
            'AverageGrad', moveOptimizer(opt.avg, @gather), ...
            'AverageSqGrad', moveOptimizer(opt.avgSq, @gather), 'Iteration', iteration, ...
            'Epoch', epoch, 'History', history(1:epoch, :), 'Best', best, ...
            'BestLoss', bestLoss, 'RandomState', rng));
    end
    if ncfg.Verbose
        fprintf(['Epoch %3d/%d: train loss %.4f | validation loss %.4f, BER %.4g, BLER %.3g' ...
            ' | lr %.2g, grad %s| %.0f s\n'], epoch, ncfg.MaxEpochs, lossSum/bitSum, ...
            vLoss, vBER, vBLER, rate, sprintf('%.3g ', median(norms, 1)), toc(clock));
    end
end
end

%% Local helpers
function [loss, grads, states] = step(params, forwardFcn, batch, trainable)
    [logits, states] = forwardFcn(params, batch, trainable);
    loss = nn.bce_loss(logits, batch.Labels, batch.Mask);
    for name = trainable
        learnables.(name{1}) = params.(name{1}).Learnables;
    end
    g = dlgradient(loss, learnables);
    for name = trainable
        grads.(name{1}) = g.(name{1});
    end
end

function [loss, ber, bler] = validate(params, forwardFcn, data, batchSize, useGPU)
    S = size(data.Slots, 3);
    lossSum = 0; errors = 0; bits = 0; blockErrors = 0;
    hasTruth = isfield(data, 'Truth');
    for first = 1:batchSize:S
        idx = first:min(first + batchSize - 1, S);
        batch = makeBatch(data, idx, useGPU);
        logits = forwardFcn(params, batch, false);
        lossSum = lossSum + double(gather(extractdata( ...
            nn.bce_loss(logits, batch.Labels, batch.Mask))))*nnz(data.Mask(:, :, :, idx));
        s = double(gather(extractdata(logits)));
        m = data.Mask(:, :, :, idx);
        labels = data.Labels(:, :, :, idx);
        errors = errors + nnz((s(m) > 0) ~= labels(m));
        bits = bits + nnz(m);
        if hasTruth && ~isempty(data.Truth(1).Code)
            for j = 1:numel(idx)
                ctx = data.Contexts(data.PilotIndex(idx(j)));
                truth = data.Truth(idx(j));
                [~, crcOK] = rx.ldpc_decode(nn.to_llr(s(:, :, :, j), ctx), truth.Code, ctx.LDPCMaxIterations);
                blockErrors = blockErrors + ~crcOK;
            end
        end
    end
    loss = lossSum/bits; ber = errors/bits;
    bler = NaN;
    if hasTruth && ~isempty(data.Truth(1).Code), bler = blockErrors/S; end
end

function batch = makeBatch(data, idx, useGPU)
    batch.Slots = data.Slots(:, :, idx);
    batch.PilotGrid = data.PilotGrid(:, :, :, idx);
    batch.Labels = data.Labels(:, :, :, idx);
    batch.Mask = single(data.Mask(:, :, :, idx));
    for f = {'Slots','PilotGrid','Labels','Mask'}
        x = batch.(f{1});
        if useGPU, x = gpuArray(x); end
        batch.(f{1}) = dlarray(x);
    end
end

function params = moveNetworks(params, operation)
    for name = fieldnames(params).'
        net = dlupdate(operation, params.(name{1}));
        state = net.State;
        state.Value = cellfun(operation, state.Value, 'UniformOutput', false);
        net.State = state;
        params.(name{1}) = net;
    end
end

function [params, opt, value, norms] = update(params, opt, forwardFcn, batch, trainable, iteration, rate, clip)
    % One Adam step on one mini-batch, with gradient clipping. norms: each
    % trained network's gradient norm before clipping, in trainable order.
    [loss, grads, states] = dlfeval(@step, params, forwardFcn, batch, trainable);
    for name = fieldnames(states).'
        params.(name{1}).State = states.(name{1});
    end
    [grads, perNetwork] = nn.clip_gradients(grads, clip);
    norms = cellfun(@(n) perNetwork.(n), trainable);
    for name = trainable
        n = name{1};
        [params.(n), opt.avg.(n), opt.avgSq.(n)] = adamupdate(params.(n), grads.(n), ...
            opt.avg.(n), opt.avgSq.(n), iteration, rate);
    end
    value = double(gather(extractdata(loss)));
end

function [result, history] = rangeTest(params, forwardFcn, trainingData, ncfg, trainable, useGPU)
    % Rate rising exponentially per step; stops once the smoothed loss is
    % 4x its minimum. Suggests a tenth of the rate at the minimum.
    if useGPU, params = moveNetworks(params, @gpuArray); end
    opt = struct();
    for name = trainable
        opt.avg.(name{1}) = []; opt.avgSq.(name{1}) = [];
    end
    rangeCfg = ncfg; rangeCfg.LearningRateSchedule = 'range';
    steps = ncfg.RangeTestSteps;
    rates = NaN(steps, 1); losses = NaN(steps, 1); smooth = NaN(steps, 1);
    norms = NaN(steps, numel(trainable));
    k = 0; epoch = 0; average = 0; diverged = false;
    while k < steps && ~diverged
        epoch = epoch + 1;
        data = trainingData(epoch);
        S = size(data.Slots, 3);
        assert(S > 0, 'The range test ran out of training slots at epoch %d.', epoch);
        order = randperm(S);
        for firstSlot = 1:ncfg.MiniBatchSize:S
            k = k + 1;
            idx = order(firstSlot:min(firstSlot + ncfg.MiniBatchSize - 1, S));
            rates(k) = nn.learning_rate(rangeCfg, k, steps);
            [params, opt, losses(k), norms(k, :)] = update(params, opt, forwardFcn, ...
                makeBatch(data, idx, useGPU), trainable, k, rates(k), ncfg.GradientClip);
            average = 0.9*average + 0.1*losses(k);
            smooth(k) = average/(1 - 0.9^k);        % Bias-corrected moving average.
            if ncfg.Verbose && (mod(k, 25) == 0 || k == 1)
                fprintf('Range test step %3d/%d: lr %.2e, smoothed loss %.4f\n', k, steps, rates(k), smooth(k));
            end
            diverged = ~isfinite(losses(k)) || smooth(k) > 4*min(smooth(1:k));
            if diverged || k == steps, break, end
        end
    end
    history = table((1:k).', rates(1:k), losses(1:k), smooth(1:k), norms(1:k, :), ...
        'VariableNames', {'Step','LearningRate','Loss','SmoothedLoss','GradientNorm'});
    [~, m] = min(smooth(1:k));
    result = struct('LearningRate', rates(m)/10, 'MinimumLossRate', rates(m), ...
        'StoppedAt', rates(k), 'Diverged', diverged, 'Steps', k);
end

function moments = moveOptimizer(moments, operation)
    % Adam moving averages: one table per network ([] before the first step).
    for name = fieldnames(moments).'
        if ~isempty(moments.(name{1}))
            moments.(name{1}) = dlupdate(operation, moments.(name{1}));
        end
    end
end
