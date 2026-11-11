function passed = test_mppa()
%% Check the 3GPP memory polynomial PA (tx.mppa, MPPA.md) and 64QAM
% Run: test_mppa
% 1. Nominal PA: the normalized coefficients reproduce the TR 38.803 model
%    (31 GHz CMOS, K = 8, M = 5) sample for sample, up to the input and
%    output scales and a fixed phase; the static curve saturates at (1, 1).
% 2. Drive: the output back-off relative to P1dB is met for -1, 3 and 7 dB,
%    and the output keeps the input RMS.
% 3. Bussgang gain: the per-subcarrier gain matches the grid estimate
%    sum(Y.*conj(X))/sum(|X|^2) and explains more than one scalar gain.
% 4. EVM falls with back-off from 0 to 12 dB, from heavy distortion to
%    the PA's most linear point.
% 5. PA bank: 30 training and 10 validation PAs, reproducible, distinct and
%    valid (saturation near 1, 1 dB compression before it).
% 6. 64QAM: rx.demap_llr equals nrSymbolDemodulate's max-log LLRs.
% 7. Link: LMMSE with the true channel (times the PA gain) decodes the
%    64QAM LDPC slot through the PA and a TDL-C channel.
passed = true;
addpath(fullfile(fileparts(mfilename('fullpath')), 'experiments'));
args = mppa_common();
base = link_config(args{2}{:}, 'SNRdB', Inf);

%% 1. Nominal PA against the TR model
fprintf('1. Nominal PA versus the TR 38.803 memory polynomial\n');
pa = tx.mppa_nominal();
s = RandStream('mt19937ar', 'Seed', 3);
x = 0.3*(randn(s, 4000, 1) + 1i*randn(s, 4000, 1))/sqrt(2);
x = x .* min(1, pa.InputScale./abs(x));          % Inside the fitted range.
yTR = polynomial(pa.Raw, x);
v = polynomial(pa.B, x/pa.InputScale);
rotation = v ./ (yTR/pa.OutputScale);
ok = max(abs(rotation - rotation(1))) < 1e-9 && abs(abs(rotation(1)) - 1) < 1e-9;
passed = report(ok, sprintf('normalized model = TR model / %.4f at input x %.4f (max deviation %.1e)', ...
    pa.OutputScale, pa.InputScale, max(abs(rotation - rotation(1))))) && passed;
c = tx.mppa_curve(pa.B);
ok = abs(c.SaturationInput - 1) < 1e-3 && abs(c.SaturationOutput - 1) < 1e-6 && ...
    abs(angle(sum(pa.B(1, :)))) < 1e-12 && c.P1dBInput < 1;
passed = report(ok, sprintf(['static curve: saturation (%.3f, %.3f), P1dB (%.3f, %.3f), ' ...
    'Psat %.2f dB above P1dB'], c.SaturationInput, c.SaturationOutput, c.P1dBInput, ...
    c.P1dBOutput, 20*log10(c.SaturationOutput/c.P1dBOutput))) && passed;

%% 2. Output back-off
fprintf('2. Output back-off relative to P1dB\n');
cfg = base;
[~, out] = tx.transmit(setfield(cfg, 'PAModel', 'none'), RandStream('mt19937ar', 'Seed', 1)); %#ok<SFLD>
for obo = [-1 3 7]
    cfg.PABackoffdB = obo;
    [y, ~, info] = tx.mppa(out.WaveformBeforePA, out.Carrier, cfg);
    rmsRatio = sqrt(mean(abs(y).^2)/mean(abs(out.WaveformBeforePA).^2));
    ok = abs(info.OBOdB - obo) < 1e-3 && abs(rmsRatio - 1) < 1e-12;
    passed = report(ok, sprintf('target %+g dB: achieved %+.4f dB, output/input RMS %.6f', ...
        obo, info.OBOdB, rmsRatio)) && passed;
end

%% 3. Per-subcarrier Bussgang gain
fprintf('3. Per-subcarrier Bussgang gain\n');
cfg = base; cfg.PABackoffdB = 3;
[w, out] = tx.transmit(cfg, RandStream('mt19937ar', 'Seed', 1));
Y = nrOFDMDemodulate(out.Carrier, w, 'Nfft', cfg.Nfft);
X = out.Grid;
measured = sum(Y.*conj(X), 2) ./ sum(abs(X).^2, 2);
e = @(g) norm(Y(:) - reshape(g.*X, [], 1))/norm(X(:));
ok = isequal(size(out.PAGain), [size(X, 1) 1]) && mean(abs(out.PAGain - measured)) < 0.05 && ...
    e(out.PAGain) < e(mean(out.PAGain));
passed = report(ok, sprintf(['F x 1 gain; mean |G - grid estimate| %.4f; residual %.3f with G, ' ...
    '%.3f with one scalar'], mean(abs(out.PAGain - measured)), e(out.PAGain), e(mean(out.PAGain)))) && passed;

%% 4. EVM against back-off
fprintf('4. EVM against output back-off\n');
backoffs = [0 3 6 12]; evm = zeros(size(backoffs));
for j = 1:numel(backoffs)
    cfg.PABackoffdB = backoffs(j);
    [w, out] = tx.transmit(cfg, RandStream('mt19937ar', 'Seed', 1));
    Y = nrOFDMDemodulate(out.Carrier, w, 'Nfft', cfg.Nfft);
    ideal = out.PAGain .* out.Grid;
    evm(j) = 100*norm(Y(out.DataMask) - ideal(out.DataMask))/norm(ideal(out.DataMask));
end
% Up to 12 dB the EVM falls; it never goes far below ~3 %: the 3GPP PA's
% small-signal gain expansion (+1.1 dB near input 0.1) distorts weak
% signals too, so EVM rises again past 12 dB (MPPA.md).
ok = all(diff(evm) < 0) && evm(1) > 8 && evm(end) < 4;
passed = report(ok, sprintf('EVM %s %% at %s dB', mat2str(round(evm, 2)), mat2str(backoffs))) && passed;

%% 5. PA bank
fprintf('5. PA bank (nn.pa_bank, dither %.2f)\n', nn_config(args{:}).PADither);
ncfg = nn_config(args{:});
bank = nn.pa_bank(base, ncfg);
again = nn.pa_bank(base, ncfg);
valid = true; sat = zeros(1, 0);
for row = [bank.Training; bank.Validation].'
    c = tx.mppa_curve(reshape(row, pa.K, pa.M));
    sat(end + 1) = c.SaturationInput; %#ok<AGROW>
    valid = valid && ~isnan(c.P1dBInput) && c.P1dBInput < c.SaturationInput;
end
ok = isequal(size(bank.Training), [30 pa.K*pa.M]) && isequal(size(bank.Validation), [10 pa.K*pa.M]) && ...
    isequal(bank, again) && isempty(intersect(bank.Training, bank.Validation, 'rows')) && ...
    valid && all(sat >= 0.8 & sat <= 1.25);
passed = report(ok, sprintf('30 + 10 distinct, reproducible, valid PAs; saturation inputs %.3f..%.3f', ...
    min(sat), max(sat))) && passed;
cfg = nn.use_pa(base, bank.Validation(1, :));
passed = report(isequal(cfg.MPCoefficients, bank.Validation(1, :)), ...
    'nn.use_pa puts a bank row into MPCoefficients') && passed;

%% 6. 64QAM demapping
fprintf('6. 64QAM soft demapping\n');
bits = randi(s, [0 1], 6*2000, 1);
sym = nrSymbolModulate(bits, '64QAM');
nu = 0.02;
r = sym + sqrt(nu/2)*(randn(s, size(sym)) + 1i*randn(s, size(sym)));
ours = rx.demap_llr(r, nu*ones(size(r)), '64QAM');
theirs = nrSymbolDemodulate(r, '64QAM', nu, 'DecisionType', 'soft');
ok = max(abs(ours - theirs)) < 1e-9*max(abs(theirs)) && tx.bits_per_symbol('64QAM') == 6;
passed = report(ok, sprintf('max |difference| %.2e against nrSymbolDemodulate', max(abs(ours - theirs)))) && passed;

%% 7. Link through the PA and a channel
fprintf('7. 64QAM LDPC link through the MP PA (4 dB back-off), TDL-C, 35 dB SNR\n');
cfg = link_config(args{2}{:}, 'PABackoffdB', 4, 'SNRdB', 35, 'Seed', 21);
[w, truth] = sim.run_link(cfg);
ctx = rx.context(cfg);
e1 = sim.evaluate(rx.receive(w, ctx, 'knownH', truth), truth, cfg);
noGain = rmfield(truth, 'PAGain');
e2 = sim.evaluate(rx.receive(w, ctx, 'knownH', noGain), truth, cfg);
ok = ~e1.BlockError && e1.UncodedErrors/e1.UncodedBits < e2.UncodedErrors/e2.UncodedBits;
passed = report(ok, sprintf('known H x PA gain: BER %.4f, CRC %s; known H alone: BER %.4f', ...
    e1.UncodedErrors/e1.UncodedBits, string(~e1.BlockError), e2.UncodedErrors/e2.UncodedBits)) && passed;

if passed, fprintf('All MP PA checks passed.\n'); else, fprintf('Some MP PA checks FAILED.\n'); end
end

%% Local helpers
function y = polynomial(A, x)
    % y(n) = sum_k sum_m A(k,m+1) x(n-m)|x(n-m)|^(k-1)
    y = zeros(size(x));
    for k = 1:size(A, 1)
        y = y + filter(A(k, :), 1, x.*abs(x).^(k-1));
    end
end

function ok = report(ok, label)
    if ok, fprintf('  PASS  %s\n', label); else, fprintf('  FAIL  %s\n', label); end
end
