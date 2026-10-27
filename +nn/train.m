function [best, history] = train(params, forwardFcn, trainingData, validation, ncfg, trainable, onBest)
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
%   ncfg        nn_config: MaxEpochs, MiniBatchSize, LearningRate, ...
%   trainable   names of the networks to update (default: all fields of
%               params); the others are frozen.
%   onBest      optional @(best, history): called whenever validation loss
%               improves, e.g. to save a checkpoint, so an interrupted run
%               keeps its best model.
%   Loss: masked binary cross-entropy (nn.bce_loss) on the coded bits, Adam.
%   Keeps the parameters with the lowest validation loss (best).
%   history: Epoch, TrainingLoss, ValidationLoss, ValidationBER,
%   ValidationBLER (NaN without Truth), Seconds.
if nargin < 6 || isempty(trainable)
    trainable = fieldnames(params).';
end
if nargin < 7
    onBest = [];
end
useGPU = ~strcmp(ncfg.ExecutionEnvironment, 'cpu') && canUseGPU;
assert(~strcmp(ncfg.ExecutionEnvironment, 'gpu') || useGPU, 'GPU requested but unavailable.');
if useGPU
    params = moveNetworks(params, @gpuArray);
end
avg = struct(); avgSq = struct();
for name = trainable
    avg.(name{1}) = []; avgSq.(name{1}) = [];
end
history = table('Size', [ncfg.MaxEpochs 6], 'VariableTypes', repmat({'double'}, 1, 6), ...
    'VariableNames', {'Epoch','TrainingLoss','ValidationLoss','ValidationBER','ValidationBLER','Seconds'});
bestLoss = Inf; iteration = 0;
for epoch = 1:ncfg.MaxEpochs
    clock = tic;
    data = trainingData(epoch);
    S = size(data.Slots, 3);
    order = randperm(S);
    lossSum = 0; bitSum = 0;
    for first = 1:ncfg.MiniBatchSize:S
        idx = order(first:min(first + ncfg.MiniBatchSize - 1, S));
        batch = makeBatch(data, idx, useGPU);
        [loss, grads, states] = dlfeval(@step, params, forwardFcn, batch, trainable);
        iteration = iteration + 1;
        for name = fieldnames(states).'
            params.(name{1}).State = states.(name{1});
        end
        for name = trainable
            n = name{1};
            [params.(n), avg.(n), avgSq.(n)] = adamupdate(params.(n), grads.(n), ...
                avg.(n), avgSq.(n), iteration, ncfg.LearningRate);
        end
        value = double(gather(extractdata(loss)));
        assert(isfinite(value), 'Non-finite training loss at iteration %d.', iteration);
        bits = nnz(data.Mask(:, :, :, idx));
        lossSum = lossSum + value*bits; bitSum = bitSum + bits;
    end
    [vLoss, vBER, vBLER] = validate(params, forwardFcn, validation, ncfg.MiniBatchSize, useGPU);
    history(epoch, :) = {epoch, lossSum/bitSum, vLoss, vBER, vBLER, toc(clock)};
    if vLoss < bestLoss
        bestLoss = vLoss;
        best.Params = moveNetworks(params, @gather); % CPU-portable.
        best.Epoch = epoch;
        if ~isempty(onBest)
            onBest(best, history(1:epoch, :));
        end
    end
    if ncfg.Verbose
        fprintf('Epoch %3d/%d: train loss %.4f | validation loss %.4f, BER %.4g, BLER %.3g | %.0f s\n', ...
            epoch, ncfg.MaxEpochs, lossSum/bitSum, vLoss, vBER, vBLER, toc(clock));
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
