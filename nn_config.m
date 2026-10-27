function ncfg = nn_config(varargin)
%NN_CONFIG Training settings for the neural receivers.
%   ncfg = nn_config() returns the defaults; nn_config('MaxEpochs',20, ...)
%   overrides named settings. The physical link stays in simo_config:
%   ncfg.Link holds simo_config name/value overrides for the base link.
%   Each training slot draws its conditions independently and uniformly
%   from the ranges below (SNR and SIR in dB); equal limits fix a value.
p = inputParser;
p.PartialMatching = false;
range = @(x) isnumeric(x) && numel(x) == 2 && isreal(x) && x(1) <= x(2) && ~any(isnan(x));
seeds = @(x) isnumeric(x) && isvector(x) && all(x >= 0 & x <= 2^32-1 & x == fix(x));
count = @(x) isnumeric(x) && isscalar(x) && x >= 1 && x == fix(x);

% Base link (simo_config overrides). LDPC so validation reports BLER; the
% network learns the coded bits on the grid either way.
addParameter(p, 'Link', {'Coding','ldpc'}, @iscell);
% Conditions drawn per slot.
addParameter(p, 'ChannelModels', {'TDL-C'}, @(x) iscellstr(x) || isstring(x));
addParameter(p, 'SNRRange', [-4 32], range);
addParameter(p, 'DopplerRange', [0 500], @(x) range(x) && all(x >= 0));
addParameter(p, 'DelaySpreadRange', [10e-9 300e-9], @(x) range(x) && all(x > 0));
addParameter(p, 'SIRRange', [0 36], @(x) isequal(x, [Inf Inf]) || range(x)); % [Inf Inf]: no interferer.
addParameter(p, 'PABackoffRange', [6 6], range); % Used when the Link sets a PAModel.
% PA bank (polynomial PA only): each slot draws one of NumTrainingPAs PAs
% whose coefficients are the Link's PACoefficients plus complex Gaussian
% dither (PADither x |coefficient|). Validation uses NumValidationPAs other
% PAs, never seen in training. PADither 0 uses the Link's PA for every slot.
addParameter(p, 'PADither', 0.1, @(x) isscalar(x) && isreal(x) && x >= 0);
addParameter(p, 'NumTrainingPAs', 30, count);
addParameter(p, 'NumValidationPAs', 10, count);
addParameter(p, 'PASeed', 2023, @(x) seeds(x) && isscalar(x));
addParameter(p, 'PilotConfigs', struct('DMRSAdditionalPosition', 0), @isstruct);
% Slots. Training slots are fresh every epoch; validation slots are fixed.
addParameter(p, 'TrainingSeeds', 42:46, seeds);
addParameter(p, 'SlotsPerSeed', 200, count);          % Per epoch.
addParameter(p, 'ValidationSeeds', 1001:1002, seeds);
addParameter(p, 'ValidationSlotsPerSeed', 50, count);
% Optimization.
addParameter(p, 'MaxEpochs', 150, count);
addParameter(p, 'MiniBatchSize', 4, count);
addParameter(p, 'LearningRate', 1e-3, @(x) isscalar(x) && x > 0);   % Peak rate.
% Learning-rate schedule (nn.learning_rate), per Adam step: a linear warmup
% from 0 over WarmupSteps, then LearningRate until DecayStart (fraction of
% all steps; decay never starts before warmup ends), then 'linear' or
% 'cosine' decay to FinalLearningRate at the last step. 'constant' never
% decays. The defaults give the old constant rate.
addParameter(p, 'LearningRateSchedule', 'constant', @(x) any(strcmp(x, {'constant','linear','cosine'})));
addParameter(p, 'WarmupSteps', 0, @(x) isscalar(x) && x >= 0 && x == fix(x));
addParameter(p, 'DecayStart', 0, @(x) isscalar(x) && x >= 0 && x <= 1);
addParameter(p, 'FinalLearningRate', 0, @(x) isscalar(x) && x >= 0);
% Scale each step's gradients (all trained networks together) down to this
% L2 norm when larger, so one bad batch (a deep fade) cannot wreck the
% networks at a high learning rate. Inf: off. history.GradientNorm logs
% the median norm before clipping.
addParameter(p, 'GradientClip', Inf, @(x) isscalar(x) && x > 0);
% Learning-rate range test instead of training (train_receiver): RangeTestSteps
% steps with the rate rising exponentially over RangeTestRates; prints the
% suggested peak LearningRate and saves models/<Name>_lr_range.mat.
addParameter(p, 'FindLearningRate', false, @(x) islogical(x) && isscalar(x));
addParameter(p, 'RangeTestSteps', 300, count);
addParameter(p, 'RangeTestRates', [1e-5 1e-1], @(x) range(x) && x(1) > 0);
addParameter(p, 'InitializationSeed', 7, @(x) seeds(x) && isscalar(x));
addParameter(p, 'ExecutionEnvironment', 'auto', @(x) any(strcmp(x, {'auto','cpu','gpu'})));
addParameter(p, 'Verbose', true, @(x) islogical(x) && isscalar(x));
% FCNN architecture: [] means the paper's Table I (see nn.fcnn).
addParameter(p, 'Widths', [], @(x) isempty(x) || (isnumeric(x) && isvector(x)));
addParameter(p, 'Dilations', [], @(x) isempty(x) || (isnumeric(x) && size(x,2) == 2));
% TCNN architecture: 'residual' (nn.tcnn, whole slot, 1-D dilated, starts as
% pass-through) or 'paper' (nn.tcnn_paper, Pihlajasalo et al. TWC 2023:
% per-symbol 2-D separable 3x3). Widths [] means 64-128-256-128-64 for both,
% with dilations 1-2-4-2-1 for 'residual' ('paper' has no dilations).
addParameter(p, 'TCNNArchitecture', 'residual', @(x) any(strcmp(x, {'residual','paper'})));
addParameter(p, 'TCNNWidths', [], @(x) isempty(x) || (isnumeric(x) && isvector(x)));
addParameter(p, 'TCNNDilations', [], @(x) isempty(x) || (isnumeric(x) && isvector(x)));
% Receiver flow (train_receiver, see nn.flow): 'fcnn', 'tcnn_feq', 'tcnn_fcnn'.
addParameter(p, 'Flow', 'fcnn', @(x) ischar(x) || isstring(x));
% Start from networks of a saved model (e.g. 'models/fcnn_pa.mat'), and
% names of networks to keep fixed (e.g. {'fcnn'}). Adam and the schedule
% start fresh: for fine-tuning or new conditions.
addParameter(p, 'InitFrom', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'Freeze', {}, @(x) iscellstr(x) || isstring(x));
% Continue an interrupted train_receiver run from models/<Name>_last.mat
% (saved every epoch): networks, Adam state, step count, schedule position,
% history and best checkpoint, and the next epoch's planned slots. Give the
% same settings as the first call; MaxEpochs may grow.
addParameter(p, 'Resume', false, @(x) islogical(x) && isscalar(x));
addParameter(p, 'Name', 'fcnn', @(x) ischar(x) || isstring(x)); % Output file name.
parse(p, varargin{:});
ncfg = p.Results;
ncfg.ChannelModels = cellstr(ncfg.ChannelModels);
ncfg.PilotConfigs = ncfg.PilotConfigs(:);
ncfg.Name = char(ncfg.Name);
ncfg.Flow = char(ncfg.Flow);
ncfg.InitFrom = char(ncfg.InitFrom);
ncfg.Freeze = cellstr(ncfg.Freeze);
ncfg.TCNNArchitecture = char(ncfg.TCNNArchitecture);
ncfg.LearningRateSchedule = char(ncfg.LearningRateSchedule);
assert(ncfg.FinalLearningRate <= ncfg.LearningRate, ...
    'FinalLearningRate must not exceed the peak LearningRate.');
assert(~(ncfg.Resume && ~isempty(ncfg.InitFrom)), ...
    'Resume continues a run; InitFrom starts a new one. Give one of them.');
if strcmp(ncfg.TCNNArchitecture, 'paper')
    assert(isempty(ncfg.TCNNDilations), 'The paper TCNN has no dilations; give TCNNWidths only.');
else
    assert(isempty(ncfg.TCNNWidths) == isempty(ncfg.TCNNDilations), ...
        'Give TCNNWidths and TCNNDilations together.');
end
assert(isempty(intersect(ncfg.TrainingSeeds, ncfg.ValidationSeeds)), ...
    'Training and validation seeds must differ.');
assert(isempty(ncfg.Widths) == isempty(ncfg.Dilations), ...
    'Give Widths and Dilations together.');
end
