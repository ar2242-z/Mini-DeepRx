function results = compare_snr(models, varargin)
%% BER and BLER versus SNR for any trained receivers, on identical slots
% Run: compare_snr({'fcnn','fcnn_pa','tcnn_feq'})
%      compare_snr({'fcnn_pa','tcnn_feq'}, 'Link',{'PABackoffdB',0}, 'Name','snr_pa0')
%      compare_snr({'fcnn','models/old/fcnn_v1.mat'}, 'SNRs',-4:2:32, 'Slots',100)
% models: cell array of names (models/<name>.mat) or .mat file paths, from
%   train_fcnn / train_receiver (any flow) or the legacy FCNN. The
%   paper's two LMMSE baselines are always added as references: 'LMMSE'
%   (LS channel estimate at the DMRS, linear interpolation, LMMSE
%   equalization) and 'LMMSE known H' (the true channel). With a PA on the
%   test link, the paper's upper bound is added too, 'Bound (known H, no
%   PA)': LMMSE with the true channel on the same slots (bits, fading,
%   noise) sent with a linear PA and a time-invariant channel (Doppler 0),
%   so there is neither PA distortion nor ICI.
% Name/value options:
%   'SNRs'   SNR points in dB of the uncoded BER panel (default 0:4:32)
%   'CodedSNRs'  SNR points in dB of the coded BER and BLER panels
%            (default 0:2:12, where LDPC goes from failing to error-free).
%            Every slot is simulated at the union of both grids.
%   'Slots'  slots per SNR point (default 30); seeds Seed, Seed+1, ...,
%            the same at every SNR and for every receiver
%   'Seed'   first slot seed (default 5001, away from training seeds)
%   'Link'   simo_config overrides of the test link. The test link starts
%            from the FIRST model's link (LinkConfig), e.g. PA model and
%            back-off, Doppler, delay spread, channel model, SIR; Coding is
%            always 'ldpc' (for BLER).
%   'UsePABank'  with a polynomial PA, cycle through the unseen validation
%            PAs saved in the models (model.PABank), so no model is tested
%            on a PA it trained on (default true)
%   'Labels' legend names (default: the model names)
%   'Name'   output name: results/<Name>.png and .mat (default 'snr_sweep')
% Every model must fit the test link (modulation, RBs, antennas, DMRS).
% Figure: uncoded BER (before LDPC), coded BER (information bits after
% LDPC decoding) and BLER (CRC failures), one panel each.
% results: table Receiver, SNRdB, BER (uncoded, before LDPC), CodedBER
% (after LDPC), BLER, BitErrors, Bits, CodedBitErrors, CodedBits,
% BlockErrors, Slots.
p = inputParser; p.PartialMatching = false;
addRequired(p, 'models', @(x) iscellstr(x) || isstring(x));
addParameter(p, 'SNRs', 0:4:32, @(x) isnumeric(x) && isvector(x));
addParameter(p, 'CodedSNRs', 0:2:12, @(x) isnumeric(x) && isvector(x));
addParameter(p, 'Slots', 30, @(x) isscalar(x) && x >= 1 && x == fix(x));
addParameter(p, 'Seed', 5001, @(x) isscalar(x) && x >= 0 && x == fix(x));
addParameter(p, 'Link', {}, @iscell);
addParameter(p, 'UsePABank', true, @(x) islogical(x) && isscalar(x));
addParameter(p, 'Labels', {}, @(x) iscellstr(x) || isstring(x));
addParameter(p, 'Name', 'snr_sweep', @(x) ischar(x) || isstring(x));
parse(p, models, varargin{:});
opt = p.Results;
snrs = union(opt.SNRs(:), opt.CodedSNRs(:)).';  % Sorted, every point simulated once.
names = cellstr(opt.models);
labels = cellstr(opt.Labels);
if isempty(labels), labels = names; end
assert(numel(labels) == numel(names), 'Give one label per model.');
root = fileparts(mfilename('fullpath'));

% Models.
loaded = cell(size(names));
for k = 1:numel(names)
    loaded{k} = loadModel(names{k}, root);
end

% Test link: the first model's link (where known), then the overrides.
quiet = {'ShowPlots',false,'RunChecks',false,'SaveData',false,'Verbose',false};
base = simo_config(quiet{:});
for k = 1:numel(loaded)
    if isfield(loaded{k}, 'LinkConfig'), base = loaded{k}.LinkConfig; break; end
end
overrides = [opt.Link, {'Coding','ldpc'}];
base = slot_config(base, struct(overrides{:}));
for k = 1:numel(loaded)
    checkLink(loaded{k}, base, names{k});
end

% PAs: the unseen validation PAs of the first model that has a bank.
pas = [];
if opt.UsePABank && strcmp(base.PAModel, 'polynomial')
    for k = 1:numel(loaded)
        if isfield(loaded{k}, 'PABank') && ~isempty(loaded{k}.PABank.Validation)
            pas = loaded{k}.PABank.Validation; break;
        end
    end
end

% Receivers: references first, then the models.
receivers = {'LMMSE', 'LMMSE known H'};
paOff = ~strcmp(base.PAModel, 'none');
if paOff, receivers{end + 1} = 'Bound (known H, no PA)'; end
refs = numel(receivers);
receivers = [receivers, labels(:).'];
R = numel(receivers); S = numel(snrs);
counts = zeros(R, S, 5);  % bit errors, bits, block errors, coded bit errors, coded bits
ctx = rx.context(base);
fprintf('%s\n', describe(base, pas));
clock = tic;
for i = 1:S
    for n = 1:opt.Slots
        cfg = base;
        cfg.SNRdB = snrs(i);
        cfg.Seed = opt.Seed + n - 1;
        if ~isempty(pas)
            cfg.PACoefficients = pas(mod(n - 1, size(pas, 1)) + 1, :);
        end
        [w, truth] = run_link(cfg);
        llrs = cell(1, R);
        llrs{1} = rx.receive(w, ctx, 'conventional');
        llrs{2} = rx.receive(w, ctx, 'knownH', truth);
        if paOff
            ideal = cfg;
            ideal.PAModel = 'none';
            ideal.MaximumDopplerShift = 0;
            [wIdeal, truthIdeal] = run_link(ideal);
            llrs{3} = rx.receive(wIdeal, ctx, 'knownH', truthIdeal);
        end
        for k = 1:numel(loaded)
            llrs{refs + k} = nn.receive(w, ctx, loaded{k});
        end
        for r = 1:R
            e = evaluate(llrs{r}, truth, cfg);
            counts(r, i, :) = squeeze(counts(r, i, :)) + [e.UncodedErrors; e.UncodedBits; e.BlockError; e.CodedErrors; e.CodedBits];
        end
    end
    line = arrayfun(@(r) sprintf('%s %.4f / %.1e / %.2f', receivers{r}, counts(r, i, 1)/counts(r, i, 2), ...
        counts(r, i, 4)/counts(r, i, 5), counts(r, i, 3)/opt.Slots), 1:R, 'UniformOutput', false);
    fprintf('SNR %5.1f dB | uncoded BER / coded BER / BLER: %s  (%.0f s)\n', snrs(i), strjoin(line, ' | '), toc(clock));
end

% Table, figure and files.
[rr, ss] = ndgrid(1:R, 1:S);
results = table(reshape(receivers(rr(:)), [], 1), reshape(snrs(ss(:)), [], 1), ...
    reshape(counts(:, :, 1)./counts(:, :, 2), [], 1), reshape(counts(:, :, 4)./counts(:, :, 5), [], 1), ...
    reshape(counts(:, :, 3)/opt.Slots, [], 1), ...
    reshape(counts(:, :, 1), [], 1), reshape(counts(:, :, 2), [], 1), ...
    reshape(counts(:, :, 4), [], 1), reshape(counts(:, :, 5), [], 1), reshape(counts(:, :, 3), [], 1), ...
    repmat(opt.Slots, R*S, 1), 'VariableNames', ...
    {'Receiver','SNRdB','BER','CodedBER','BLER','BitErrors','Bits','CodedBitErrors','CodedBits', ...
    'BlockErrors','Slots'});
folder = fullfile(root, 'results');
if ~isfolder(folder), mkdir(folder); end
name = char(opt.Name);
conditions = describe(base, pas);
models = names; %#ok<NASGU>
save(fullfile(folder, [name '.mat']), 'results', 'conditions', 'models', 'opt');
f = figure('Visible', 'off', 'Position', [0 0 1600 430]);
tiledlayout(1, 3);
styles = {'o-', 's-', '^-', 'v-', 'd-', 'p-', 'h-', '>-', '<-'};
for panel = 1:3
    nexttile;
    % Uncoded panel on SNRs, coded panels on CodedSNRs.
    if panel == 1, x = sort(opt.SNRs(:)).'; else, x = sort(opt.CodedSNRs(:)).'; end
    at = ismember(snrs, x);
    for r = 1:R
        switch panel
            case 1, y = squeeze(counts(r, at, 1)./counts(r, at, 2));
            case 2, y = squeeze(counts(r, at, 4)./counts(r, at, 5));
            case 3, y = squeeze(counts(r, at, 3))/opt.Slots;
        end
        y(y == 0) = NaN;  % Zero errors: no point on a log axis.
        semilogy(x, y, styles{mod(r - 1, numel(styles)) + 1}, 'LineWidth', 1.2, ...
            'DisplayName', receivers{r}); hold on
    end
    grid on; xlabel('SNR (dB)');
    if numel(x) > 1, xlim(x([1 end])); end  % Same axis even where errors run out.
    if panel == 1
        ylabel('BER'); title('Uncoded BER (before LDPC)');
    elseif panel == 2
        % Down to one bit error in all the slots of a point.
        ylabel('BER'); title('Coded BER (information bits after LDPC)');
        ylim([0.5/max(counts(:, :, 5), [], 'all') 0.5]);
    else
        ylabel('BLER'); title(sprintf('LDPC BLER (%d slots per point)', opt.Slots));
        ylim([min(0.5/opt.Slots, 0.1) 1]);
    end
    legend('Location', 'southwest', 'Interpreter', 'none');
end
sgtitle(conditions, 'FontSize', 9, 'Interpreter', 'none');
exportgraphics(f, fullfile(folder, [name '.png']), 'Resolution', 120);
close(f);
fprintf('Saved %s.png and %s.mat in %s (no point = no errors).\n', name, name, folder);
end

%% Local helpers
function model = loadModel(name, root)
    file = name;
    if ~endsWith(file, '.mat'), file = fullfile(root, 'models', [name '.mat']); end
    assert(isfile(file), 'No model file %s.', file);
    s = load(file, 'model');
    model = s.model;
    if ~isfield(model, 'FeatureLayout') && ~isfield(model, 'Flow') && isfield(model, 'ScoreConvention')
        % Legacy network.m model: same features and output as deeprx-v1.
        assert(strcmp(model.ScoreConvention, 'positive means bit 1'));
        model.FeatureLayout = 'deeprx-v1';
    end
end

function checkLink(model, cfg, name)
    if ~isfield(model, 'LinkConfig'), return; end
    fields = {'Modulation','NumResourceBlocks','SubcarrierSpacing','Nfft','NumReceiveAntennas', ...
        'DMRSTypeAPosition','DMRSLength','DMRSAdditionalPosition','DMRSConfigurationType', ...
        'NumCDMGroupsWithoutData'};
    for f = fields
        a = model.LinkConfig.(f{1}); b = cfg.(f{1});
        assert(isequal(a, b), '%s was trained with %s = %s, but the test link has %s.', ...
            name, f{1}, mat2str(a), mat2str(b));
    end
end

function text = describe(cfg, pas)
    pa = 'no PA';
    if ~strcmp(cfg.PAModel, 'none')
        pa = sprintf('%s PA at %g dB back-off', cfg.PAModel, cfg.PABackoffdB);
        if ~isempty(pas), pa = sprintf('%s (%d unseen PAs)', pa, size(pas, 1)); end
    end
    sir = 'no interferer';
    if isfinite(cfg.SIRdB), sir = sprintf('SIR %g dB', cfg.SIRdB); end
    text = sprintf('%s, %g ns, %g Hz, %s, LDPC %g, %s, %s', cfg.ChannelModel, cfg.DelaySpread*1e9, ...
        cfg.MaximumDopplerShift, cfg.Modulation, cfg.TargetCodeRate, pa, sir);
end
