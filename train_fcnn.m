function [model, history] = train_fcnn(varargin)
%% Train the FCNN (DeepRx) receiver
% Run: train_fcnn                        % defaults from nn_config
%      train_fcnn('MaxEpochs',20,'ChannelModels',{'TDL-A','TDL-C','TDL-E'})
%      addpath experiments; args = pa_common; train_fcnn(args{:},'Name','fcnn_pa')  % PA on
% Name/value pairs are nn_config settings. Steps:
%   1. plan disjoint validation and training slots (nn.plan_slots)
%   2. simulate the fixed validation set once (nn.make_data, with truth
%      for BLER); training slots are simulated fresh every epoch
%   3. build the network (nn.fcnn) and train it (nn.train)
%   4. save the best validation checkpoint to models/<Name>.mat, updated
%      every time validation improves, so an interrupted run keeps it
% The model is used with nn.receive(rxWaveform, rx.context(cfg), model).
ncfg = nn_config(varargin{:});
quiet = {'ShowPlots',false,'RunChecks',false,'SaveData',false,'Verbose',false};
baseCfg = simo_config(quiet{:}, ncfg.Link{:});
ctx = rx.context(baseCfg);

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
trainingPlan = nn.plan_slots(ncfg, ncfg.TrainingSeeds, ncfg.SlotsPerSeed, ncfg.MaxEpochs, ...
    [roots; validationPlan.Seed], paBank.Training);
fprintf('Simulating %d validation slots...\n', height(validationPlan));
validation = nn.make_data(baseCfg, ncfg, validationPlan, true);
trainingData = @(epoch) nn.make_data(baseCfg, ncfg, trainingPlan(trainingPlan.Epoch == epoch, :));

% 3. Network and training.
inputSize = [ctx.NumSubcarriers ctx.NumSymbols 2*(2*baseCfg.NumReceiveAntennas + 1)];
rng(ncfg.InitializationSeed, 'twister');
if isempty(ncfg.Widths)
    net = nn.fcnn(inputSize, ctx.BitsPerSymbol);
else
    net = nn.fcnn(inputSize, ctx.BitsPerSymbol, ncfg.Widths, ncfg.Dilations);
end
fprintf('FCNN: %d learnable parameters; %d training slots per epoch.\n', ...
    sum(cellfun(@numel, net.Learnables.Value)), sum(trainingPlan.Epoch == 1));
folder = fullfile(fileparts(mfilename('fullpath')), 'models');
if ~isfolder(folder), mkdir(folder); end
file = fullfile(folder, [ncfg.Name '.mat']);
fprintf('Checkpoints go to %s whenever validation loss improves.\n', file);
save_model = @(best, history) saveModel(file, best, history, inputSize, baseCfg, ncfg, validationPlan, paBank);
[best, history] = nn.train(struct('fcnn', net), ...
    @(p, b, t) nn.fcnn_forward(p, b, t, ctx), trainingData, validation, ncfg, [], save_model);

% 4. The best checkpoint is already saved; save once more with the full history.
model = saveModel(file, best, history, inputSize, baseCfg, ncfg, validationPlan, paBank);
fprintf('Saved %s (best validation epoch %d).\n', file, best.Epoch);
end

function model = saveModel(file, best, history, inputSize, baseCfg, ncfg, validationPlan, paBank)
    % Everything needed to use the network again with nn.receive.
    model.Net = best.Params.fcnn;
    model.InputSize = inputSize;
    model.FeatureLayout = 'deeprx-v1';
    model.Output = 'logit of P(bit = 1) per bit plane (nn.to_llr: LLR = -logit)';
    model.LinkConfig = baseCfg;
    model.TrainingConfig = ncfg;
    model.BestEpoch = best.Epoch;
    model.ValidationPlan = validationPlan;
    model.PABank = paBank; % Training and (unseen) validation PAs; empty without a bank.
    save(file, 'model', 'history');
end
