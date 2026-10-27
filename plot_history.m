function summary = plot_history(models, varargin)
%% Training curves of saved models: loss, validation BER and BLER per epoch
% Run: plot_history({'fcnn','fcnn_pa','tcnn_feq'})
%      plot_history({'fcnn','fcnn_quick'}, 'Labels',{'150 epochs','20 epochs'}, 'Name','fcnn_runs')
% models: cell array of names (models/<name>.mat) or .mat file paths saved
%   by train_fcnn / train_receiver, which store the per-epoch history next
%   to the model.
% Panels:
%   1. training loss (dashed) and validation loss (solid), one colour per
%      model: a widening gap means overfitting; both still falling means
%      more epochs would help
%   2. validation BER before LDPC (log scale)
%   3. validation BLER after LDPC
%   The best epoch (lowest validation loss, the saved network) is circled.
% A run still in progress saves its history up to its latest best epoch,
% so its curves end there.
% Name/value options: 'Labels' (legend names), 'Name' (results/<Name>.png,
% default 'training_history').
% summary: table Model, Flow, Epochs, BestEpoch, TrainingLoss and
% ValidationLoss / BER / BLER at the best epoch, Hours.
p = inputParser; p.PartialMatching = false;
addRequired(p, 'models', @(x) iscellstr(x) || isstring(x));
addParameter(p, 'Labels', {}, @(x) iscellstr(x) || isstring(x));
addParameter(p, 'Name', 'training_history', @(x) ischar(x) || isstring(x));
parse(p, models, varargin{:});
names = cellstr(p.Results.models);
labels = cellstr(p.Results.Labels);
if isempty(labels), labels = names; end
assert(numel(labels) == numel(names), 'Give one label per model.');
root = fileparts(mfilename('fullpath'));

f = figure('Visible', 'off', 'Position', [0 0 1350 420]);
tiledlayout(1, 3);
ax = arrayfun(@(k) nexttile, 1:3);
colours = lines(numel(names));
rows = cell(numel(names), 9);
for k = 1:numel(names)
    file = names{k};
    if ~endsWith(file, '.mat'), file = fullfile(root, 'models', [names{k} '.mat']); end
    assert(isfile(file), 'No model file %s.', file);
    s = load(file);
    assert(isfield(s, 'history'), '%s has no training history.', file);
    h = s.history(s.history.Epoch > 0, :);
    best = s.model.BestEpoch;
    b = find(h.Epoch == best, 1);
    c = colours(k, :);
    flow = 'fcnn'; if isfield(s.model, 'Flow'), flow = s.model.Flow; end

    plot(ax(1), h.Epoch, h.TrainingLoss, '--', 'Color', c, 'DisplayName', [labels{k} ' training']); hold(ax(1), 'on');
    plot(ax(1), h.Epoch, h.ValidationLoss, '-', 'Color', c, 'LineWidth', 1.2, 'DisplayName', [labels{k} ' validation']);
    plot(ax(1), best, h.ValidationLoss(b), 'o', 'Color', c, 'MarkerSize', 8, 'HandleVisibility', 'off');
    semilogy(ax(2), h.Epoch, h.ValidationBER, '-', 'Color', c, 'LineWidth', 1.2, 'DisplayName', labels{k}); hold(ax(2), 'on');
    plot(ax(2), best, h.ValidationBER(b), 'o', 'Color', c, 'MarkerSize', 8, 'HandleVisibility', 'off');
    plot(ax(3), h.Epoch, h.ValidationBLER, '-', 'Color', c, 'LineWidth', 1.2, 'DisplayName', labels{k}); hold(ax(3), 'on');
    plot(ax(3), best, h.ValidationBLER(b), 'o', 'Color', c, 'MarkerSize', 8, 'HandleVisibility', 'off');

    rows(k, :) = {labels{k}, flow, height(h), best, h.TrainingLoss(b), h.ValidationLoss(b), ...
        h.ValidationBER(b), h.ValidationBLER(b), sum(h.Seconds)/3600};
end
set(ax(2), 'YScale', 'log');
titles = {'Loss (dashed: training, solid: validation)', 'Validation BER (before LDPC)', 'Validation BLER (LDPC)'};
ylabels = {'bit cross-entropy', 'BER', 'BLER'};
for i = 1:3
    grid(ax(i), 'on'); xlabel(ax(i), 'epoch'); ylabel(ax(i), ylabels{i}); title(ax(i), titles{i});
    legend(ax(i), 'Location', 'northeast', 'Interpreter', 'none');
end
sgtitle('Training history (circle: best epoch, the saved network)', 'FontSize', 10);
folder = fullfile(root, 'results');
if ~isfolder(folder), mkdir(folder); end
out = fullfile(folder, [char(p.Results.Name) '.png']);
exportgraphics(f, out, 'Resolution', 120);
close(f);

summary = cell2table(rows, 'VariableNames', {'Model','Flow','Epochs','BestEpoch', ...
    'TrainingLoss','ValidationLoss','ValidationBER','ValidationBLER','Hours'});
disp(summary);
fprintf('Saved %s.\n', out);
end
