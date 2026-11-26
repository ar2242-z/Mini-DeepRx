function out = tmp_simo_vs_mimo(varargin)
%TMP_SIMO_VS_MIMO Temporary test: SIMO 1x2 versus SU-MIMO 2x2, coded BLER and goodput.
%   out = tmp_simo_vs_mimo()            full sweep (background run)
%   out = tmp_simo_vs_mimo('Quick',true) tiny grid, for a smoke test
%
%   Links (all LDPC, TDL-C 300 ns, 2 Rx, 26 RB, 15 kHz, 1 ms slot, no PA,
%   no interferer, DMRS type 1 with DMRSLength 1 + DMRSAdditionalPosition 1
%   = two pilot symbols, NumCDMGroupsWithoutData 2 for every link, so every
%   link has 12 x 312 = 3744 data REs per layer):
%     SIMO 64QAM R=1/2   port 0        TBS 11272
%     MIMO 64QAM R=1/2   ports 0/2     TBS 22536  (same modulation)
%     MIMO 16QAM R=3/8   ports 0/2     TBS 11272  (same payload as SIMO)
%   Doppler 0 Hz and 100 Hz. Equal total Tx power (tx.transmit scales by
%   1/sqrt(Nt)) and equal N0 (cfg.SNRdB).
%   Receivers on the same slots:
%     'conv (DMRS timing)'     LS -> interpolation -> LMMSE, DMRS timing
%     'conv (perfect timing)'  the same with the true timing (isolates the
%                              channel estimation)
%     'knownH (genie)'         true channel and true timing (rx.receive
%                              forces perfect timing in this mode)
%   Stopping, per receiver and SNR point: TargetErrors block errors or
%   MaxSlots slots (at least MinSlots). A receiver's sweep ends after TWO
%   consecutive points with BLER < StopBLER.
%   Slot seeds Seed, Seed+1, ... are the same for every link and SNR: this
%   aids reproducibility, but with different Nt and modulation it does NOT
%   give identical bits, noise or channel realizations across links.
%   BLER confidence: 95% Clopper-Pearson. SNR at 10% BLER: linear
%   interpolation of log10(BLER) in SNR at the first crossing; its interval
%   is the same crossing on the upper and lower CI curves.
%   Goodput = (1 - BLER) * TBS / 1 ms (no retransmission overhead).
%   Limitation of the conventional baseline: rx.equalize's nu omits the
%   channel-estimation error, so its LLRs are overconfident and its measured
%   performance may be worse than a receiver that accounts for it.
p = inputParser; p.PartialMatching = false;
addParameter(p, 'Quick', false);
addParameter(p, 'TargetErrors', 100);
addParameter(p, 'MaxSlots', 1000);
addParameter(p, 'MinSlots', 50);
addParameter(p, 'StopBLER', 0.01);
addParameter(p, 'Step', 1);
addParameter(p, 'MaxSNR', 40);
addParameter(p, 'Seed', 7001);
addParameter(p, 'Name', 'tmp_simo_vs_mimo');
parse(p, varargin{:});
opt = p.Results;
if opt.Quick
    opt.TargetErrors = 3; opt.MaxSlots = 6; opt.MinSlots = 2; opt.Step = 6; opt.MaxSNR = 30;
    opt.Name = [opt.Name '_quick'];
end

common = {'Coding','ldpc','ChannelModel','TDL-C','DelaySpread',300e-9, ...
    'NumReceiveAntennas',2,'NumResourceBlocks',26,'SubcarrierSpacing',15, ...
    'DMRSConfigurationType',1,'DMRSLength',1,'DMRSAdditionalPosition',1, ...
    'NumCDMGroupsWithoutData',2,'PAModel','none','SIRdB',Inf, ...
    'LDPCMaxIterations',25,'TimingEstimation','dmrs'};
mimo = {'NumTransmitAntennas',2,'NumLayers',2,'DMRSPortSet',[0 2]};
links = struct( ...
    'Name',  {'SIMO 1x2 64QAM R1/2', 'MIMO 2x2 64QAM R1/2', 'MIMO 2x2 16QAM R3/8'}, ...
    'Args',  {{'Modulation','64QAM','TargetCodeRate',0.5,'DMRSPortSet',0}, ...
              [{'Modulation','64QAM','TargetCodeRate',0.5}, mimo], ...
              [{'Modulation','16QAM','TargetCodeRate',0.375}, mimo]}, ...
    'StartSNR', {4, 10, 4});
dopplers = [0 100];
receivers = {'conv (DMRS timing)', 'conv (perfect timing)', 'knownH (genie)'};

% Payload check before anything runs.
tbs = zeros(1, numel(links));
for k = 1:numel(links)
    cfg = link_config(common{:}, links(k).Args{:}, 'SNRdB', Inf, 'Seed', opt.Seed);
    [~, truth] = sim.run_link(cfg);
    tbs(k) = numel(truth.InfoBits);
    fprintf('%-22s TBS %6d  coded bits %6d  effective rate %.4f  data REs/layer %d\n', ...
        links(k).Name, tbs(k), numel(truth.CodedBits), tbs(k)/numel(truth.CodedBits), ...
        numel(truth.CodedBits)/cfg.NumLayers/tx.bits_per_symbol(cfg.Modulation));
end
assert(abs(tbs(3)/tbs(1) - 1) < 0.02, 'Same-payload pair does not match: %d vs %d.', tbs(1), tbs(3));

[li, di] = ndgrid(1:numel(links), 1:numel(dopplers));
jobs = numel(li);
parts = cell(1, jobs);
clock = tic;
parfor j = 1:jobs
    link = links(li(j));
    base = link_config(common{:}, link.Args{:}, 'MaximumDopplerShift', dopplers(di(j)));
    parts{j} = sweep(base, link, tbs(li(j)), dopplers(di(j)), receivers, opt);
end
results = vertcat(parts{:});
fprintf('Sweep done in %.1f min.\n', toc(clock)/60);

% SNR at 10% BLER.
summary = table();
for k = 1:numel(links)
    for d = dopplers
        for r = 1:numel(receivers)
            s = results(strcmp(results.Link, links(k).Name) & results.DopplerHz == d & ...
                strcmp(results.Receiver, receivers{r}), :);
            row = table(string(links(k).Name), d, string(receivers{r}), tbs(k), ...
                crossing(s.SNRdB, s.BLER, s.Slots), crossing(s.SNRdB, s.CIHigh, s.Slots), ...
                crossing(s.SNRdB, s.CILow, s.Slots), max(s.GoodputMbps), ...
                'VariableNames', {'Link','DopplerHz','Receiver','TBS','SNR10','SNR10Upper', ...
                'SNR10Lower','PeakGoodputMbps'});
            summary = [summary; row]; %#ok<AGROW>
        end
    end
end
disp(summary);

root = fileparts(mfilename('fullpath'));
folder = fullfile(root, 'results');
if ~isfolder(folder), mkdir(folder); end
save(fullfile(folder, [opt.Name '.mat']), 'results', 'summary', 'opt', 'tbs', 'common');
plotAll(results, links, dopplers, receivers, opt, fullfile(folder, [opt.Name '.png']));
fprintf('Saved results/%s.mat and .png\n', opt.Name);
out.results = results; out.summary = summary; out.tbs = tbs;
end

function T = sweep(base, link, tbs, doppler, receivers, opt)
R = numel(receivers);
ctxDmrs = rx.context(base);
ctxPerfect = ctxDmrs; ctxPerfect.TimingEstimation = 'perfect';
active = true(1, R);
below = zeros(1, R);          % Consecutive points with BLER < StopBLER.
rows = {};
snr = link.StartSNR;
while any(active) && snr <= opt.MaxSNR
    errs = zeros(1, R); slots = zeros(1, R); bitErr = zeros(1, R); bits = zeros(1, R);
    need = active;
    for n = 1:opt.MaxSlots
        cfg = base; cfg.SNRdB = snr; cfg.Seed = opt.Seed + n - 1;
        [w, truth] = sim.run_link(cfg);
        for r = find(need)
            switch r
                case 1, llr = rx.receive(w, ctxDmrs, 'conventional');
                case 2, llr = rx.receive(w, ctxPerfect, 'conventional', truth);
                case 3, llr = rx.receive(w, ctxDmrs, 'knownH', truth);
            end
            e = sim.evaluate(llr, truth, cfg);
            errs(r) = errs(r) + e.BlockError; slots(r) = slots(r) + 1;
            bitErr(r) = bitErr(r) + e.UncodedErrors; bits(r) = bits(r) + e.UncodedBits;
            if errs(r) >= opt.TargetErrors && slots(r) >= opt.MinSlots, need(r) = false; end
        end
        if ~any(need), break; end
    end
    for r = find(active)
        bler = errs(r)/slots(r);
        [lo, hi] = clopperPearson(errs(r), slots(r));
        rows(end+1, :) = {link.Name, doppler, receivers{r}, snr, slots(r), errs(r), bler, lo, hi, ...
            bitErr(r)/bits(r), tbs, (1 - bler)*tbs/1e-3/1e6}; %#ok<AGROW>
        if bler < opt.StopBLER, below(r) = below(r) + 1; else, below(r) = 0; end
        if below(r) >= 2, active(r) = false; end
    end
    fprintf('[%s, %d Hz] SNR %4.1f dB | %s\n', link.Name, doppler, snr, strjoin(arrayfun(@(r) ...
        sprintf('%s %d/%d', receivers{r}, errs(r), slots(r)), 1:R, 'UniformOutput', false), ' | '));
    snr = snr + opt.Step;
end
T = cell2table(rows, 'VariableNames', {'Link','DopplerHz','Receiver','SNRdB','Slots', ...
    'BlockErrors','BLER','CILow','CIHigh','UncodedBER','TBS','GoodputMbps'});
end

function [lo, hi] = clopperPearson(k, n)
a = 0.05;
if k == 0, lo = 0; else, lo = betainv(a/2, k, n - k + 1); end
if k == n, hi = 1; else, hi = betainv(1 - a/2, k + 1, n - k); end
end

function x = crossing(snr, bler, slots)
% First SNR where the curve drops from >= 0.1 to < 0.1, interpolating log10(BLER).
x = NaN;
y = log10(max(bler, 0.5./slots));    % Zero errors: half an error, for the log.
t = log10(0.1);
for i = 1:numel(snr) - 1
    if y(i) >= t && y(i+1) < t
        x = snr(i) + (t - y(i))/(y(i+1) - y(i))*(snr(i+1) - snr(i));
        return;
    end
end
if ~isempty(y) && y(1) < t, x = -Inf; end   % Already below at the first point.
end

function plotAll(results, links, dopplers, receivers, opt, file)
f = figure('Visible', 'off', 'Position', [0 0 1500 900]);
tiledlayout(2, 2);
colors = lines(numel(links));
styles = {'-o', '--s', ':^'};
for d = 1:numel(dopplers)
    for panel = 1:2
        nexttile; hold on; grid on;
        for k = 1:numel(links)
            for r = 1:numel(receivers)
                s = results(strcmp(results.Link, links(k).Name) & results.DopplerHz == dopplers(d) & ...
                    strcmp(results.Receiver, receivers{r}), :);
                if panel == 1
                    y = s.BLER; y(y == 0) = NaN;
                    errorbar(s.SNRdB, y, y - s.CILow, s.CIHigh - y, styles{r}, 'Color', colors(k, :), ...
                        'LineWidth', 1.1, 'DisplayName', sprintf('%s, %s', links(k).Name, receivers{r}));
                else
                    plot(s.SNRdB, s.GoodputMbps, styles{r}, 'Color', colors(k, :), 'LineWidth', 1.1, ...
                        'DisplayName', sprintf('%s, %s', links(k).Name, receivers{r}));
                end
            end
        end
        xlabel('SNR (dB)');
        if panel == 1
            set(gca, 'YScale', 'log'); ylim([1e-3 1]); yline(0.1, 'k:', 'HandleVisibility', 'off');
            ylabel('BLER'); title(sprintf('BLER, %d Hz Doppler (95%% CI)', dopplers(d)));
            legend('Location', 'southwest', 'FontSize', 7, 'Interpreter', 'none');
        else
            ylabel('Goodput (Mbit/s)'); title(sprintf('Goodput (1-BLER) TBS / 1 ms, %d Hz', dopplers(d)));
            legend('Location', 'northwest', 'FontSize', 7, 'Interpreter', 'none');
        end
    end
end
sgtitle(sprintf(['TDL-C 300 ns, 2 Rx, 26 RB, 2 DMRS symbols, CDM groups w/o data 2, LDPC, ' ...
    'no PA. Stop: %d errors or %d slots'], opt.TargetErrors, opt.MaxSlots), 'FontSize', 9);
exportgraphics(f, file, 'Resolution', 120);
close(f);
end
