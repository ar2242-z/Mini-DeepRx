function [model, history] = train_receiver(varargin)
%% Train a neural receiver flow (FCNN, TCNN/FEQ, TCNN+FCNN hybrid)
% Run: train_receiver('Flow','fcnn')                     % same as train_fcnn
%      addpath experiments; args = pa_common;
%      train_receiver(args{:},'Flow','tcnn_feq','Name','tcnn_feq')
%      train_receiver(args{:},'Flow','tcnn_fcnn','TCNNArchitecture','paper','Name','hybrid_paper')
%      train_receiver(args{:},'Flow','tcnn_fcnn','Name','tcnn_fcnn', ...
%          'InitFrom','models/fcnn_pa.mat','Freeze',{'fcnn'})
%   With a learning-rate schedule (experiments/pa_schedule):
%      args = pa_schedule;
%      train_receiver(args{:},'Flow','tcnn_feq','Name','tcnn_feq_s','FindLearningRate',true)
%      train_receiver(args{:},'Flow','tcnn_feq','Name','tcnn_feq_s')
%      train_receiver(args{:},'Flow','tcnn_feq','Name','tcnn_feq_s','Resume',true)
% Name/value pairs are nn_config settings; 'Flow' picks the chain (nn.flow).
% Start: from scratch (default), from a saved model's networks ('InitFrom',
% fresh Adam and schedule) or where an interrupted run stopped ('Resume',
% from models/<Name>_last.mat, saved after every epoch).
% 'FindLearningRate',true runs only the range test (nn.train) and saves
% models/<Name>_lr_range.mat; model is then the result, history its steps.
% Steps:
%   1. plan disjoint validation and training slots (nn.plan_slots), with
%      separate training and validation PAs when the Link has a PA bank
%   2. simulate the fixed validation set once (nn.make_data, with truth
%      for BLER); training slots are simulated fresh every epoch
%   3. build the flow's networks (nn.fcnn, nn.tcnn), optionally load some
%      from a saved model (InitFrom) and keep some fixed (Freeze)
%   4. train (nn.train) and save the best validation checkpoint to
%      models/<Name>.mat, updated every time validation improves
% The model is used with nn.receive(rxWaveform, rx.context(cfg), model).
ncfg = nn_config(varargin{:});
flow = nn.flow(ncfg.Flow);
quiet = {'ShowPlots',false,'RunChecks',false,'SaveData',false,'Verbose',false};
baseCfg = simo_config(quiet{:}, ncfg.Link{:});
ctx = rx.context(baseCfg);
if flow.SinglePilotLayout
    % The conventional blocks take the pilot layout from ctx.
    assert(isscalar(ncfg.PilotConfigs), 'Flow %s needs a single PilotConfig.', flow.Name);
    ctx = rx.context(slot_config(baseCfg, ncfg.PilotConfigs));
end

% 1-2. Slots: validation first, so training can exclude its seeds.
% With a polynomial PA, training and validation slots use different PAs.
paBank = nn.pa_bank(baseCfg, ncfg);
if ~isempty(paBank.Training)
    fprintf('PA bank: %d training PAs, %d unseen validation PAs, back-off %g to %g dB.\n', ...
        size(paBank.Training, 1), size(paBank.Validation, 1), ncfg.PABackoffRange);
end
roots = [ncfg.TrainingSeeds(:); ncfg.ValidationSeeds(:)];
validationPlan = nn.plan_slots(ncfg, ncfg.ValidationSeeds, ncfg.ValidationSlotsPerSeed, 1, roots, ...
    paBank.Validation);
planEpochs = ncfg.MaxEpochs;
if ncfg.FindLearningRate   % Enough slots for every range-test step.
    stepsPerEpoch = ceil(numel(ncfg.TrainingSeeds)*ncfg.SlotsPerSeed/ncfg.MiniBatchSize);
    planEpochs = max(planEpochs, ceil(ncfg.RangeTestSteps/stepsPerEpoch));
end
trainingPlan = nn.plan_slots(ncfg, ncfg.TrainingSeeds, ncfg.SlotsPerSeed, planEpochs, ...
    [roots; validationPlan.Seed], paBank.Training);
fprintf('Simulating %d validation slots...\n', height(validationPlan));
validation = nn.make_data(baseCfg, ncfg, validationPlan, true);
trainingData = @(epoch) nn.make_data(baseCfg, ncfg, trainingPlan(trainingPlan.Epoch == epoch, :));

% 3. Networks: fresh (FCNN first, so an FCNN starts as in train_fcnn),
% then any loaded from InitFrom; Freeze keeps networks fixed.
inputSize = [ctx.NumSubcarriers ctx.NumSymbols 2*(2*baseCfg.NumReceiveAntennas + 1)];
tcnnInputSize = nn.tcnn_input_size(ctx, ncfg.TCNNArchitecture);
rng(ncfg.InitializationSeed, 'twister');
params = struct();
if ismember('fcnn', flow.Networks)
    if isempty(ncfg.Widths)
        params.fcnn = nn.fcnn(inputSize, ctx.BitsPerSymbol);
    else
        params.fcnn = nn.fcnn(inputSize, ctx.BitsPerSymbol, ncfg.Widths, ncfg.Dilations);
    end
end
if ismember('tcnn', flow.Networks)
    if strcmp(ncfg.TCNNArchitecture, 'paper')
        if isempty(ncfg.TCNNWidths)
            params.tcnn = nn.tcnn_paper(tcnnInputSize);
        else
            params.tcnn = nn.tcnn_paper(tcnnInputSize, ncfg.TCNNWidths);
        end
    elseif isempty(ncfg.TCNNWidths)
        params.tcnn = nn.tcnn(tcnnInputSize);
    else
        params.tcnn = nn.tcnn(tcnnInputSize, ncfg.TCNNWidths, ncfg.TCNNDilations);
    end
end
if ismember('calibration', flow.Networks)
    params.calibration = nn.llr_calibration();
end
if ~isempty(ncfg.InitFrom)
    loaded = nn.model_nets(getfield(load(ncfg.InitFrom, 'model'), 'model')); %#ok<GFLD>
    for name = intersect(fieldnames(loaded).', flow.Networks)
        n = name{1};
        assert(isequal(loaded.(n).Layers(1).InputSize, params.(n).Layers(1).InputSize), ...
            'The %s network in %s has input %s; this link needs %s.', n, ncfg.InitFrom, ...
            mat2str(loaded.(n).Layers(1).InputSize), mat2str(params.(n).Layers(1).InputSize));
        params.(n) = loaded.(n);
        fprintf('Loaded the %s network from %s.\n', n, ncfg.InitFrom);
    end
end
assert(all(ismember(ncfg.Freeze, flow.Networks)), 'Freeze names networks that flow %s does not use.', flow.Name);
trainable = setdiff(flow.Networks, ncfg.Freeze, 'stable');
assert(~isempty(trainable), 'Every network is frozen; nothing to train.');
for name = flow.Networks
    state = 'trained'; if ~ismember(name{1}, trainable), state = 'frozen'; end
    fprintf('%s: %d learnable parameters (%s).\n', upper(name{1}), ...
        sum(cellfun(@numel, params.(name{1}).Learnables.Value)), state);
end
fprintf('Flow %s; %d training slots per epoch.\n', flow.Name, sum(trainingPlan.Epoch == 1));
folder = fullfile(fileparts(mfilename('fullpath')), 'models');
if ~isfolder(folder), mkdir(folder); end
file = fullfile(folder, [ncfg.Name '.mat']);
lastFile = fullfile(folder, [ncfg.Name '_last.mat']);
forward = @(p, b, t) flow.Forward(p, b, t, ctx);
if ncfg.FindLearningRate
    % Range test only: no validation, no model saved.
    [result, steps] = nn.train(params, forward, trainingData, validation, ncfg, trainable);
    rangeFile = fullfile(folder, [ncfg.Name '_lr_range.mat']);
    save(rangeFile, 'result', 'steps');
    fprintf(['Range test: lowest loss at lr %.2g; stopped at %.2g after %d steps.\n' ...
        'Suggested peak LearningRate %.2g (saved %s).\n'], result.MinimumLossRate, ...
        result.StoppedAt, result.Steps, result.LearningRate, rangeFile);
    model = result; history = steps;
    return
end
start = [];
if ncfg.Resume
    assert(isfile(lastFile), 'Resume needs %s from an earlier run.', lastFile);
    saved = load(lastFile, 'state', 'ncfg');
    checkResume(saved.ncfg, ncfg);
    start = saved.state;
    fprintf('Resuming %s.\n', lastFile);
end
fprintf('Checkpoints go to %s whenever validation loss improves;\n', file);
fprintf('%s after every epoch (for Resume).\n', lastFile);
info = struct('Flow', flow.Name, 'Trainable', {trainable}, 'InputSize', inputSize, ...
    'TCNNInputSize', tcnnInputSize, 'TCNNArchitecture', ncfg.TCNNArchitecture, 'LinkConfig', baseCfg, 'TrainingConfig', ncfg, ...
    'ValidationPlan', validationPlan, 'PABank', paBank);
save_model = @(best, history) saveModel(file, best, history, info);
[best, history] = nn.train(params, forward, trainingData, validation, ncfg, trainable, ...
    save_model, @(state) saveState(lastFile, state, ncfg), start);

% 4. The best checkpoint is already saved; save once more with the full history.
model = saveModel(file, best, history, info);
fprintf('Saved %s (best validation epoch %d).\n', file, best.Epoch);
end

function model = saveModel(file, best, history, info)
    % Everything needed to use the networks again with nn.receive.
    model.Flow = info.Flow;
    model.Nets = best.Params;                 % One field per network.
    if isfield(best.Params, 'fcnn')
        model.Net = best.Params.fcnn;         % FCNN models as from train_fcnn.
        model.InputSize = info.InputSize;
        model.FeatureLayout = 'deeprx-v1';
    end
    if isfield(best.Params, 'tcnn')
        model.TCNNInputSize = info.TCNNInputSize;
        model.TCNNArchitecture = info.TCNNArchitecture;
    end
    model.Output = 'logit of P(bit = 1) per bit plane (nn.to_llr: LLR = -logit)';
    model.Trainable = info.Trainable;
    model.LinkConfig = info.LinkConfig;
    model.TrainingConfig = info.TrainingConfig;
    model.BestEpoch = best.Epoch;
    model.ValidationPlan = info.ValidationPlan;
    model.PABank = info.PABank;
    save(file, 'model', 'history');
end

function saveState(file, state, ncfg) %#ok<INUSD> (saved by name)
    % Everything nn.train needs to continue (Resume), written to a temporary
    % file first so an interruption mid-save cannot corrupt the last state.
    temporary = [file '.tmp.mat'];
    save(temporary, 'state', 'ncfg');
    movefile(temporary, file, 'f');
end

function checkResume(saved, ncfg)
    % Settings that define the run's slots, steps and networks must match;
    % MaxEpochs may grow, the rest (rates, Verbose, ...) may change.
    fields = {'Flow','Link','ChannelModels','SNRRange','DopplerRange','DelaySpreadRange', ...
        'SIRRange','PABackoffRange','PADither','NumTrainingPAs','NumValidationPAs','PASeed', ...
        'PilotConfigs','TrainingSeeds','SlotsPerSeed','ValidationSeeds', ...
        'ValidationSlotsPerSeed','MiniBatchSize','Widths','Dilations','TCNNArchitecture', ...
        'TCNNWidths','TCNNDilations','Freeze'};
    differ = fields(~cellfun(@(f) isequal(saved.(f), ncfg.(f)), fields));
    assert(isempty(differ), 'Resume needs the same settings as the first run; these differ: %s.', ...
        strjoin(differ, ', '));
    assert(ncfg.MaxEpochs >= saved.MaxEpochs, 'Resume cannot lower MaxEpochs (%d).', saved.MaxEpochs);
    if ncfg.MaxEpochs > saved.MaxEpochs && ~strcmp(ncfg.LearningRateSchedule, 'constant')
        warning('MaxEpochs grew from %d to %d: the learning-rate decay stretches from here on.', ...
            saved.MaxEpochs, ncfg.MaxEpochs);
    end
end
