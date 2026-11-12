function [xhat, nu, info] = dpoi(Y, H, ctx, options)
%DPOI Digital Post-Inverse: equalize, then undo the transmit PA in time.
%   [xhat, nu, info] = rx.dpoi(Y, H, ctx)
%   [xhat, nu, info] = rx.dpoi(Y, H, ctx, options)
%   Y, H: F x N x Nr (one slot). xhat, nu: D x 1 at the data REs in
%   transmit order, as rx.equalize returns them, for rx.demap_llr.
%   The one-shot receiver-side PA compensation of Babaroglu et al.,
%   "Cellular Digital Post-Distortion: Signal Processing Methods and RF
%   Measurements", IEEE WAMICON 2023 (Sec. II-B, II-C):
%     1. Linear equalization of the whole grid, the antennas combined by
%        maximum ratio: Q = sum conj(H).*Y / (sum |H|^2 + sigma^2). This is
%        the LMMSE equalizer, which holds back the noise on faded
%        subcarriers; the nonlinear step below would spread it over the
%        whole symbol. Its per-subcarrier gain mu = g/(g + sigma^2) is
%        divided out again after step 4.
%     2. Each OFDM symbol to time at L times the sample rate (an L*Nfft
%        IFFT of the occupied subcarriers): q(n).
%     3. Memory polynomial post-inverse (paper eq. 6):
%            x(n) = sum_p sum_d c(p,d) q(n-d) |q(n-d)|^(p-1)
%        with delays in oversampled samples, circular within the symbol
%        (exact for PA memory shorter than the cyclic prefix).
%     4. FFT back to the occupied subcarriers.
%     5. Coefficients c: least squares so that the output equals the known
%        pilots (eq. 9), once per slot, with no knowledge of the PA.
%   Differences from the paper, which has a DMRS-only symbol:
%     - Here the DMRS symbol also carries data on the other subcarriers.
%       The basis functions come from the received symbol, so only the
%       target needs known values: the fit is made in frequency, on the
%       pilot subcarriers only. The symbol's envelope is that of a data
%       symbol, so the paper's DMRS boosting is not needed.
%     - The fit weights each pilot by its channel gain (its noise variance
%       after equalization is sigma^2/gain) and is pulled slightly towards
%       "no correction" (options.Ridge), since the basis functions are
%       strongly correlated.
%     - Noise in q biases a least-squares inverse towards a smaller output
%       (the paper notes this). The output is rescaled so the pilots come
%       out with gain 1, which the demapper needs for 16QAM and 64QAM.
%   nu: the error power left at the pilots after the correction (noise plus
%   residual distortion, sigma^2/gain per RE), so the LLRs match what is
%   actually left.
%   options (struct, all optional):
%     Orders        polynomial orders p (default [1 3 5 7], the paper's P = 7)
%     Memory        largest delay D in oversampled samples (default 4)
%     Oversampling  L (default 4)
%     Ridge         pull towards no correction, relative (default 1e-2)
%     NoiseVariance sigma^2 of the equalizer in step 1 (default 0: zero
%                   forcing; rx.receive passes rx.ls_estimate's value)
%     Fit           true (default); false applies no correction, leaving
%                   the equalizer and the noise estimate only
%     Target        F x N grid of known symbols to fit on instead of the
%                   pilots (zeros = unknown). The true grid gives the
%                   genie coefficients: the best this post-inverse can do.
%     ChannelDof    parameters the channel estimate fitted to the same
%                   pilots (rx.delay_estimate's dof times the pilot
%                   symbols), so the residual noise estimate is not
%                   biased low
%   info: Coefficients (orders x delays), Bias, Sigma2, Equalized and
%   Corrected (F x N grids before and after the post-inverse).
if nargin < 4, options = struct(); end
opt = struct('Orders', [1 3 5 7], 'Memory', 4, 'Oversampling', 4, 'Ridge', 1e-2, ...
    'Fit', true, 'Target', [], 'ChannelDof', 0, 'NoiseVariance', 0);
for name = fieldnames(options).'
    assert(isfield(opt, name{1}), 'Unknown DPoI option %s.', name{1});
    opt.(name{1}) = options.(name{1});
end
assert(size(Y, 4) == 1, 'rx.dpoi works on one slot at a time.');
assert(ctx.NumLayers == 1 && size(H, 5) == 1, 'rx.dpoi handles one layer only.');
assert(opt.Orders(1) == 1, 'The first order must be 1 (the linear term).');
F = ctx.NumSubcarriers; N = ctx.NumSymbols;

% 1. Linear equalization of the whole grid (Q is the unbiased output).
gain = max(sum(real(H).^2 + imag(H).^2, 3), eps);   % F x N
Q = sum(conj(H) .* Y, 3) ./ gain;

% Known symbols to fit on: the pilots, or a given grid.
if isempty(opt.Target)
    rows = ctx.PilotIndices(:);
    target = ctx.PilotGrid(rows);
else
    rows = find(opt.Target ~= 0);
    target = opt.Target(rows);
end
weight = gain(rows) / mean(gain(rows));

% 2-4. Basis functions on the grid: column j is the post-inverse output for
% one (order, delay) term alone. A delay of d oversampled samples is a
% phase ramp on the subcarriers, so each order needs one IFFT/FFT pair.
J = 1;
if opt.Fit
    M = opt.Oversampling * ctx.Nfft;
    k = (0:F-1).' - F/2;
    bins = mod(k, M) + 1;
    Z = complex(zeros(M, N));
    mu = gain ./ (gain + opt.NoiseVariance);
    Z(bins, :) = Q .* mu;
    q = ifft(Z) * (M/sqrt(F));              % About unit RMS for unit-power symbols.
    delays = 0:opt.Memory;
    J = numel(opt.Orders) * numel(delays);
    basis = complex(zeros(F*N, J));
    for i = 1:numel(opt.Orders)
        T = fft(q .* abs(q).^(opt.Orders(i) - 1));
        T = T(bins, :) * (sqrt(F)/M);
        for d = delays
            j = (d * numel(opt.Orders)) + i;   % Orders x delays, column-major.
            basis(:, j) = reshape(T .* exp(-2i*pi*k*d/M) ./ mu, [], 1);
        end
    end

    % 5. Weighted ridge least squares at the known symbols, in scaled
    % coordinates, pulled towards c = [1 0 0 ...] (output = input).
    A = basis(rows, :);
    scale = sqrt(mean(abs(A).^2, 1));
    As = A ./ scale;
    c0 = zeros(J, 1); c0(1) = scale(1);
    G = As' * (weight .* As);
    rhs = As' * (weight .* (target - As*c0));
    c = (c0 + (G + opt.Ridge*real(trace(G))/J*eye(J)) \ rhs) ./ scale.';
    out = A*c;
else
    c = 1;
    out = Q(rows);
end

% Remove the bias: the known symbols must come out with gain 1.
bias = sum(weight .* conj(target) .* out) / sum(weight .* abs(target).^2);
c = c / bias;
out = out / bias;
if opt.Fit
    corrected = reshape(basis*c, F, N);
else
    corrected = Q / bias;
end

% Error left at the known symbols: sigma^2/gain per RE.
dof = min(J + opt.ChannelDof, 0.9*numel(rows));
sigma2 = sum(gain(rows) .* abs(out - target).^2) / (numel(rows) - dof);
xhat = corrected(ctx.DataIndices);
nu = sigma2 ./ gain(ctx.DataIndices);

if opt.Fit
    info.Coefficients = reshape(c, numel(opt.Orders), []);
else
    info.Coefficients = c;
end
info.Bias = bias;
info.Sigma2 = sigma2;
info.Equalized = Q;
info.Corrected = corrected;
end
