function summary = plot_mppa(varargin)
%% The 3GPP memory polynomial PA: Fig. 4 of Pihlajasalo et al. (TWC 2023)
% Run: plot_mppa
%      plot_mppa('Slots',5, 'Name','mppa_fig4')
% Uses the shared MP PA experiment settings (experiments/mppa_common.m) and
% its 10 unseen validation PAs (nn.pa_bank), see MPPA.md.
% Figure 1 (results/<Name>.png), like the paper's Fig. 4:
%   1. static AM/AM of each validation PA (normalized input and output;
%      held flat past saturation), the nominal 3GPP PA dashed
%   2. static AM/AM's phase: AM/PM in radians
%   3. EVM of the PA output against the output back-off relative to P1dB,
%      0..12 dB, averaged over Slots slots (14 OFDM symbols each) per PA.
%      EVM is measured on the data REs after the per-subcarrier Bussgang
%      gain (tx.mppa), with no channel or noise.
% Figure 2 (results/<Name>_memory.png): instantaneous output amplitude and
%   phase against input amplitude for the nominal PA at 3 dB back-off, at
%   the PA's own (oversampled) rate. With memory the curve becomes a cloud
%   around the static curve (TR 38.803 Fig. A.1-12).
% Name/value options: 'Backoffs' (dB, default 0:12), 'Slots' (per PA and
% back-off, default 3), 'Seed' (first slot seed, default 9001), 'Name'
% (default 'mppa_fig4').
% summary: table PA (0 nominal, 1..10 validation), SaturationOutput,
% P1dBOutput, PsatOverP1dB (dB), AMPMAtP1dB (deg), and EVM (%) at each
% back-off.
p = inputParser; p.PartialMatching = false;
addParameter(p, 'Backoffs', 0:12, @(x) isnumeric(x) && isvector(x));
addParameter(p, 'Slots', 3, @(x) isscalar(x) && x >= 1 && x == fix(x));
addParameter(p, 'Seed', 9001, @(x) isscalar(x) && x >= 0 && x == fix(x));
addParameter(p, 'Name', 'mppa_fig4', @(x) ischar(x) || isstring(x));
parse(p, varargin{:});
opt = p.Results;
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'experiments'));

args = mppa_common();
base = simo_config(args{2}{:}, 'SNRdB', Inf);
bank = nn.pa_bank(base, nn_config(args{:}));
nominal = tx.mppa_nominal();
pas = [nominal.B(:).'; bank.Validation];     % Row 1: nominal.
P = size(pas, 1); O = numel(opt.Backoffs);

% Static curves and EVM.
u = (0:1e-3:1.2).';
amam = zeros(numel(u), P); ampm = amam; stats = zeros(P, 4);
evm = zeros(P, O);
clock = tic;
for i = 1:P
    B = reshape(pas(i, :), nominal.K, nominal.M);
    c = tx.mppa_curve(B, u);
    held = u > c.SaturationInput;
    amam(:, i) = c.AMAM; amam(held, i) = c.SaturationOutput;
    ampm(:, i) = c.AMPM; ampm(held, i) = c.AMPM(find(~held, 1, 'last'));
    stats(i, :) = [c.SaturationOutput, c.P1dBOutput, 20*log10(c.SaturationOutput/c.P1dBOutput), ...
        rad2deg(c.AMPM(u == c.P1dBInput))];
    cfg = base;
    cfg.MPCoefficients = pas(i, :);
    for j = 1:O
        cfg.PABackoffdB = opt.Backoffs(j);
        err = 0; ref = 0;
        for n = 1:opt.Slots
            [w, out] = tx.transmit(cfg, RandStream('mt19937ar', 'Seed', opt.Seed + n - 1));
            Y = nrOFDMDemodulate(out.Carrier, w, 'Nfft', cfg.Nfft);
            ideal = out.PAGain .* out.Grid;
            err = err + sum(abs(Y(out.DataMask) - ideal(out.DataMask)).^2);
            ref = ref + sum(abs(ideal(out.DataMask)).^2);
        end
        evm(i, j) = 100*sqrt(err/ref);
    end
    fprintf('PA %2d: EVM %.1f %% at %g dB ... %.2f %% at %g dB  (%.0f s)\n', i - 1, ...
        evm(i, 1), opt.Backoffs(1), evm(i, end), opt.Backoffs(end), toc(clock));
end

% Colours (dataviz reference palette): validation PAs share one hue, the
% nominal PA is the primary ink, the 3GPP 64QAM EVM limit is muted.
blue = [42 120 214]/255; ink = [11 11 11]/255; muted = [82 81 78]/255;
gridColour = [0.88 0.88 0.87];
f = figure('Visible', 'off', 'Position', [0 0 1350 420], 'Color', 'w');
tiledlayout(1, 3, 'TileSpacing', 'compact');
ax = arrayfun(@(k) nexttile, 1:3);
hold(ax, 'on');
for i = 2:P
    plot(ax(1), u, amam(:, i), 'Color', [blue 0.55], 'LineWidth', 1.2);
    plot(ax(2), u, ampm(:, i), 'Color', [blue 0.55], 'LineWidth', 1.2);
    plot(ax(3), opt.Backoffs, evm(i, :), 'Color', [blue 0.55], 'LineWidth', 1.2);
end
hv = plot(ax(1), NaN, NaN, 'Color', blue, 'LineWidth', 1.2);
hn = plot(ax(1), u, amam(:, 1), '--', 'Color', ink, 'LineWidth', 1.6);
plot(ax(2), u, ampm(:, 1), '--', 'Color', ink, 'LineWidth', 1.6);
plot(ax(3), opt.Backoffs, evm(1, :), '--', 'Color', ink, 'LineWidth', 1.6);
yline(ax(3), 8, ':', '64QAM limit 8 %', 'Color', muted, 'LabelHorizontalAlignment', 'right', ...
    'LabelVerticalAlignment', 'top', 'FontSize', 9);
legend(ax(1), [hv hn], {sprintf('%d validation PAs', P - 1), 'Nominal 3GPP PA'}, ...
    'Location', 'southeast', 'Box', 'off');
title(ax(1), 'AM-AM response'); xlabel(ax(1), 'Input amplitude'); ylabel(ax(1), 'Output amplitude');
title(ax(2), 'AM-PM response'); xlabel(ax(2), 'Input amplitude'); ylabel(ax(2), 'Phase response (rad)');
title(ax(3), 'Average EVMs'); xlabel(ax(3), 'PA output backoff wrt. P_{1dB} (dB)'); ylabel(ax(3), 'EVM (%)');
xlim(ax(1), [0 u(end)]); xlim(ax(2), [0 u(end)]); xlim(ax(3), opt.Backoffs([1 end]));
ylim(ax(1), [0 1.2]); ylim(ax(3), [0 ceil(max(evm(:))/2)*2]);
set(ax, 'Box', 'off', 'XGrid', 'on', 'YGrid', 'on', 'GridColor', gridColour, 'GridAlpha', 1, ...
    'XColor', muted, 'YColor', muted, 'FontSize', 10);
for a = ax, a.Title.Color = ink; end
sgtitle(f, sprintf(['3GPP TR 38.803 31 GHz CMOS memory polynomial PA (K = 8, M = 5, %dx ' ...
    'oversampled), dither %.2f'], base.PAOversampling, nn_config(args{:}).PADither), 'FontSize', 11);

out = fullfile(root, 'results');
if ~isfolder(out), mkdir(out); end
name = char(opt.Name);
exportgraphics(f, fullfile(out, [name '.png']), 'Resolution', 150);
close(f);

% Memory: instantaneous response of the nominal PA at 3 dB back-off.
cfg = base; cfg.PABackoffdB = 3;
[~, out1] = tx.transmit(cfg, RandStream('mt19937ar', 'Seed', opt.Seed));
[~, ~, info] = tx.mppa(out1.WaveformBeforePA, out1.Carrier, cfg);
uIn = info.Input; vOut = info.Output;
pick = 1:7:numel(uIn);
g = figure('Visible', 'off', 'Position', [0 0 900 380], 'Color', 'w');
tiledlayout(1, 2, 'TileSpacing', 'compact');
b1 = nexttile; hold(b1, 'on');
scatter(b1, abs(uIn(pick)), abs(vOut(pick)), 4, blue, 'filled', 'MarkerFaceAlpha', 0.25);
plot(b1, u, amam(:, 1), '--', 'Color', ink, 'LineWidth', 1.6);
title(b1, 'AM-AM with memory'); xlabel(b1, 'Input amplitude'); ylabel(b1, 'Output amplitude');
xlim(b1, [0 1.2]); ylim(b1, [0 1.2]);
b2 = nexttile; hold(b2, 'on');
phase = angle(vOut(pick) .* conj(uIn(pick)));
scatter(b2, abs(uIn(pick)), phase, 4, blue, 'filled', 'MarkerFaceAlpha', 0.25);
plot(b2, u, ampm(:, 1), '--', 'Color', ink, 'LineWidth', 1.6);
title(b2, 'AM-PM with memory'); xlabel(b2, 'Input amplitude'); ylabel(b2, 'Phase response (rad)');
xlim(b2, [0 1.2]); ylim(b2, [-pi/2 pi/2]);
legend(b1, {'Samples (3 dB back-off)', 'Static curve'}, 'Location', 'southeast', 'Box', 'off');
set([b1 b2], 'Box', 'off', 'XGrid', 'on', 'YGrid', 'on', 'GridColor', gridColour, 'GridAlpha', 1, ...
    'XColor', muted, 'YColor', muted, 'FontSize', 10);
exportgraphics(g, fullfile(out, [name '_memory.png']), 'Resolution', 150);
close(g);

summary = array2table([(0:P-1).', stats, evm], 'VariableNames', [{'PA','SaturationOutput', ...
    'P1dBOutput','PsatOverP1dB','AMPMAtP1dB'}, compose('EVM_%gdB', opt.Backoffs)]);
save(fullfile(out, [name '.mat']), 'summary', 'pas', 'u', 'amam', 'ampm', 'evm', 'opt');
fprintf('Saved %s.png, %s_memory.png and %s.mat in results/\n', name, name, name);
end
