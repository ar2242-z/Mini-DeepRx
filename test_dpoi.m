function passed = test_dpoi()
%% Check the Digital Post-Inverse baseline (rx.dpoi, rx.delay_estimate)
% Run: test_dpoi        (CPU only, about a minute)
% 1. Channel smoothing: rx.delay_estimate is closer to the true channel
%    than LS with linear interpolation, without and with a PA.
% 2. Why DPoI needs it: with per-pilot LS the equalized pilots equal the
%    transmitted pilots exactly, so they show no distortion to fit.
% 3. The post-inverse is the paper's memory polynomial: the frequency-domain
%    implementation (one FFT pair per order, delays as phase ramps) equals
%    eq. (6) computed sample by sample in time.
% 4. No PA, no noise: DPoI changes nothing (coefficients [1 0 0 ...]).
% 5. PA, no noise, true channel: DPoI removes a large part of the distortion, for
%    the polynomial PA and the 3GPP memory polynomial PA (not all: samples
%    clipped at saturation cannot be restored), and the pilot fit
%    is close to the genie fit on the whole grid.
% 6. The noise estimate matches the error actually left on the data.
% 7. Link: with a PA at high SNR, DPoI has a lower uncoded BER than the
%    same receiver without the post-inverse ('smoothed'); without a PA it
%    does not lose much.
% 8. compare_snr adds the DPoI baselines.
passed = true;
quiet = {'ShowPlots',false,'RunChecks',false,'SaveData',false,'Verbose',false};
static = {'MaximumDopplerShift',0};
poly = {'PAModel','polynomial','PABackoffdB',3};
mppa = {'SubcarrierSpacing',60,'Nfft',1024,'NumResourceBlocks',66,'Modulation','64QAM', ...
    'PAModel','mp3gpp','PAOversampling',4,'PABackoffdB',3};

%% 1. Channel smoothing
fprintf('1. Delay-domain channel smoothing against linear interpolation\n');
cases = {'no PA', {}; 'polynomial PA, 3 dB back-off', poly; '3GPP MP PA, 3 dB back-off', mppa};
for c = 1:size(cases, 1)
    cfg = simo_config(quiet{:}, static{:}, cases{c, 2}{:}, 'SNRdB', 20, 'TimingEstimation', 'perfect');
    ctx = rx.context(cfg);
    e = zeros(1, 2);
    for seed = 31:34
        cfg.Seed = seed;
        [w, truth] = run_link(cfg);
        Y = rx.ofdm_demod(rx.sync(w, ctx, truth.TimingOffset), ctx);
        Hp = rx.ls_estimate(Y, ctx);
        Ht = truth.H .* truth.PAGain;
        nmse = @(H) sum(abs(H - Ht).^2, 'all') / sum(abs(Ht).^2, 'all');
        e = e + [nmse(rx.interpolate(Hp, ctx)), nmse(rx.delay_estimate(Hp, ctx))]/4;
    end
    passed = report(e(2) < 0.6*e(1), sprintf('%s: NMSE %.1f dB linear, %.1f dB smoothed', ...
        cases{c, 1}, 10*log10(e(1)), 10*log10(e(2)))) && passed;
end

%% 2. Per-pilot LS hides the distortion
fprintf('2. Equalized pilots with per-pilot LS and with the smoothed channel\n');
cfg = simo_config(quiet{:}, static{:}, poly{:}, 'SNRdB', 30, 'Seed', 31);
ctx = rx.context(cfg);
[w, truth] = run_link(cfg);
[~, tr] = rx.receive(w, ctx, 'smoothed');
equalized = @(H) sum(conj(H).*tr.Y, 3) ./ sum(abs(H).^2, 3);
pilotError = @(H) rms(subsref(equalized(H) - ctx.PilotGrid, substruct('()', {ctx.PilotIndices})));
eLS = pilotError(rx.interpolate(tr.Hp, ctx));
eSmooth = pilotError(tr.H);
passed = report(eLS < 1e-10 && eSmooth > 0.05, sprintf(['pilot error %.1e with per-pilot LS (nothing ' ...
    'to fit), %.3f with the smoothed channel'], eLS, eSmooth)) && passed;

%% 3. The post-inverse is eq. (6)
fprintf('3. Frequency-domain implementation against the time-domain memory polynomial\n');
options = struct('Orders', [1 3 5 7], 'Memory', 4, 'Oversampling', 4);
[~, ~, info] = rx.dpoi(tr.Y, tr.H, ctx, options);
direct = postInverse(info.Equalized, info.Coefficients, options, ctx);
d = max(abs(direct - info.Corrected), [], 'all');
passed = report(d < 1e-9 && isequal(size(info.Coefficients), [4 5]), ...
    sprintf('4 orders x 5 delays; max |difference| %.1e', d)) && passed;

%% 4. No PA, no noise
fprintf('4. No PA and no noise: no correction\n');
cfg = simo_config(quiet{:}, static{:}, 'SNRdB', Inf, 'Seed', 31);
ctx = rx.context(cfg);
[w, truth] = run_link(cfg);
[~, tr] = rx.receive(w, ctx, 'dpoi_knownH', truth);
c = tr.DPoI.Coefficients;
rest = c; rest(1) = 0;
x = truth.Grid(ctx.DataIndices);
ok = abs(c(1) - 1) < 1e-3 && max(abs(rest(:))) < 1e-3 && rms(tr.Xhat - x) < 1e-3;
passed = report(ok, sprintf('c(1,0) = %.5f, largest other |c| %.1e, data error %.1e', ...
    abs(c(1)), max(abs(rest(:))), rms(tr.Xhat - x))) && passed;

%% 5. PA, no noise, true channel
fprintf('5. PA without noise, true channel: EVM before and after the post-inverse\n');
cases = {'polynomial PA', poly, 0.7; '3GPP MP PA', mppa, 0.7};
for c = 1:size(cases, 1)
    cfg = simo_config(quiet{:}, static{:}, cases{c, 2}{:}, 'SNRdB', Inf, 'Seed', 31);
    ctx = rx.context(cfg);
    [w, truth] = run_link(cfg);
    x = truth.Grid(ctx.DataIndices);
    [~, a] = rx.receive(w, ctx, 'knownH', truth);
    [~, b] = rx.receive(w, ctx, 'dpoi_knownH', truth);
    genie = struct('Target', truth.Grid);
    [xg, ~] = rx.dpoi(b.Y, b.H, ctx, genie);
    evm = 100*[rms(a.Xhat - x), rms(b.Xhat - x), rms(xg - x)];
    ok = evm(2) < cases{c, 3}*evm(1) && evm(2) < 1.25*evm(3);
    passed = report(ok, sprintf('%s: EVM %.1f %% -> %.1f %% (genie fit %.1f %%)', ...
        cases{c, 1}, evm)) && passed;
end

%% 6. Noise estimate
fprintf('6. Noise estimate against the error left on the data (20 dB, polynomial PA)\n');
cfg = simo_config(quiet{:}, static{:}, poly{:}, 'SNRdB', 20);
ctx = rx.context(cfg);
ratio = zeros(1, 2);
for seed = 31:36
    cfg.Seed = seed;
    [w, truth] = run_link(cfg);
    x = truth.Grid(ctx.DataIndices);
    [~, a] = rx.receive(w, ctx, 'dpoi');
    [~, b] = rx.receive(w, ctx, 'smoothed');
    ratio = ratio + [mean(abs(a.Xhat - x).^2 ./ a.Nu), mean(abs(b.Xhat - x).^2 ./ b.Nu)]/6;
end
passed = report(all(ratio > 0.7 & ratio < 1.5), sprintf(['mean |error|^2 / nu: %.2f with DPoI, ' ...
    '%.2f smoothed (1 = calibrated)'], ratio)) && passed;

%% 7. Link
fprintf('7. Uncoded BER, static TDL-C, 6 slots\n');
cases = {'polynomial PA, 30 dB', [poly, {'SNRdB',30}], 0.5; ...
    '3GPP MP PA 64QAM, 30 dB', [mppa, {'SNRdB',30}], 0.8; ...
    'no PA, 15 dB', {'SNRdB',15}, 1.25};
for c = 1:size(cases, 1)
    cfg = simo_config(quiet{:}, static{:}, cases{c, 2}{:});
    ctx = rx.context(cfg);
    errors = zeros(1, 3); bits = 0;
    modes = {'conventional', 'smoothed', 'dpoi'};
    for seed = 41:46
        cfg.Seed = seed;
        [w, truth] = run_link(cfg);
        for m = 1:3
            errors(m) = errors(m) + nnz((rx.receive(w, ctx, modes{m}) < 0) ~= truth.CodedBits);
        end
        bits = bits + numel(truth.CodedBits);
    end
    ber = errors/bits;
    passed = report(ber(3) <= cases{c, 3}*ber(2), sprintf(['%s: BER %.4f conventional, %.4f smoothed, ' ...
        '%.4f DPoI'], cases{c, 1}, ber)) && passed;
end

%% 8. compare_snr
fprintf('8. compare_snr with the DPoI baselines\n');
root = fileparts(mfilename('fullpath'));
name = 'test_dpoi_tmp';
evalc(['results = compare_snr({}, ''Link'', poly, ''SNRs'', 30, ''CodedSNRs'', 30, ''Slots'', 1, ' ...
    '''Baselines'', {''smoothed'',''dpoi'',''dpoi_knownH'',''dpoi_genie''}, ''Name'', name);']);
for ext = {'.png', '.mat'}
    file = fullfile(root, 'results', [name ext{1}]);
    if isfile(file), delete(file); end
end
names = unique(results.Receiver, 'stable').';
expected = {'LMMSE', 'LMMSE known H', 'Bound (known H, no PA)', 'LMMSE smoothed', 'DPoI', ...
    'DPoI known H', 'DPoI genie fit'};
passed = report(isequal(names, expected) && all(isfinite(results.BER)), ...
    sprintf('receivers: %s', strjoin(names, ', '))) && passed;

if passed, fprintf('All DPoI checks passed.\n'); else, fprintf('Some DPoI checks FAILED.\n'); end
end

%% Local helpers
function X = postInverse(Q, c, options, ctx)
    % Paper eq. (6) sample by sample: x(n) = sum c(p,d) q(n-d)|q(n-d)|^(p-1)
    % on the L-times oversampled symbols, delays circular.
    [F, N] = size(Q);
    M = options.Oversampling * ctx.Nfft;
    bins = mod((0:F-1).' - F/2, M) + 1;
    Z = complex(zeros(M, N));
    Z(bins, :) = Q;
    q = ifft(Z) * (M/sqrt(F));
    x = zeros(M, N);
    for i = 1:numel(options.Orders)
        term = q .* abs(q).^(options.Orders(i) - 1);
        for d = 0:options.Memory
            x = x + c(i, d + 1) * circshift(term, d, 1);
        end
    end
    X = fft(x);
    X = X(bins, :) * (sqrt(F)/M);
end

function ok = report(ok, label)
    if ok, fprintf('  PASS  %s\n', label); else, fprintf('  FAIL  %s\n', label); end
end
