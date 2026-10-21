function [results, sigma2Hat] = decode_simo_ls(data)
%% SIMO receiver: known H versus LS H from one pilot-bearing OFDM symbol
% First run mini_simo_tdl.m, then run this file.
% With no input: load the saved slot, report, check, save, and plot.
% With a data structure containing X,H,Y,meta: return results quietly.
% The sweep uses this second form to reuse exactly the same receiver.
% Arrays use (frequency,time,receive antenna), matching 5G Toolbox.
% Equations (1)-(3) in DeepRx: Y=H*X+noise, pilot LS, then LMMSE.
% No LDPC is used: the reported BER compares uncoded data bits
% (16QAM by default; see Modulation in simo_config).

%% 1. Load the experiment
folder = fileparts(mfilename('fullpath'));
interactive = nargin == 0;
if interactive
inputFile = fullfile(folder, 'mini_simo_tdl_data.mat');
assert(isfile(inputFile), 'Run mini_simo_tdl.m first to generate the data.');
data = load(inputFile, 'X', 'Y', 'H', 'meta');
end
Y = data.Y;
pilotMask = data.meta.PilotMask;
dataIndices = data.meta.DataIndices;
assert(isfield(data.meta,'Modulation'), ...
    'Saved slot predates the Modulation setting; rerun mini_simo_tdl.');
modulation = data.meta.Modulation;
% In a real receiver, pilots are generated from the known configuration.
% Here copy ONLY known pilot symbols; unknown data symbols stay zero.
pilotGrid = zeros(size(pilotMask));
pilotGrid(pilotMask) = data.X(pilotMask);

%% 2. Estimate H and noise power using only received pilots
[Hls, sigma2Hat, pilotInfo] = estimateFromPilots(Y, pilotGrid, pilotMask);
% sigma2Hat estimates E[|noise|^2], the COMPLEX noise variance per antenna.
% sigmaHat is its square root, not the standard deviation of one real part.
sigmaHat = sqrt(sigma2Hat);
if interactive
fprintf('Pilot-bearing OFDM symbol (MATLAB index): %d\n', pilotInfo.Symbol);
fprintf('Pilots per antenna: %d\n', numel(pilotInfo.Subcarriers));
fprintf('Estimated sigma^2: %.6g; estimated sigma: %.6g\n', sigma2Hat, sigmaHat);
end

%% 3. Equalize with known H and with the one-pilot LS estimate
% Both branches use the SAME pilot-only noise estimate. Neither reads the
% simulator noise variance, actual noise samples, SNR, or noiseless Y.
XhatKnown = equalizeSIMO(Y, data.H, sigma2Hat);
XhatLS = equalizeSIMO(Y, Hls, sigma2Hat);
bitsKnown = demapHard(XhatKnown(dataIndices), modulation);
bitsLS = demapHard(XhatLS(dataIndices), modulation);

%% 4. Evaluate against transmitted bits (used only here, not in detection)
referenceBits = data.meta.Bits;
numBits = numel(referenceBits);
errorsKnown = nnz(bitsKnown ~= referenceBits);
errorsLS = nnz(bitsLS ~= referenceBits);
berKnown = errorsKnown / numBits;
berLS = errorsLS / numBits;
results = table({'Known H'; 'LS H, one pilot symbol'}, ...
    [errorsKnown; errorsLS], [numBits; numBits], [berKnown; berLS], ...
    'VariableNames', {'Receiver','BitErrors','TotalBits','BER'});
if interactive
disp(results);
fprintf('One-slot result: zero errors does not imply zero long-run BER.\n');

% Channel NMSE is a diagnostic; true H is not used to obtain Hls or sigma2Hat.
channelNMSE = sum(abs(Hls(:)-data.H(:)).^2) / sum(abs(data.H(:)).^2);
fprintf('LS channel normalized mean-square error: %.6g\n', channelNMSE);
checkReceiver(modulation);
save(fullfile(folder,'decode_simo_ls_results.mat'), ...
    'Hls','sigma2Hat','sigmaHat','pilotInfo','XhatKnown','XhatLS', ...
    'bitsKnown','bitsLS','results','channelNMSE');

%% 5. Plot channel estimates, equalized symbols, and measured BER
plotReceiver(data.H, Hls, XhatKnown(dataIndices), XhatLS(dataIndices), ...
    [berKnown berLS], [errorsKnown errorsLS], numBits, modulation);
end
end

%% Local helper functions
function [Hls, sigma2Hat, details] = estimateFromPilots(Y, pilots, mask)
    pilotSymbols = find(any(mask,1));
    assert(isscalar(pilotSymbols), ...
        'This introductory receiver expects one pilot-bearing OFDM symbol.');
    n = pilotSymbols(1);
    f = find(mask(:,n));
    assert(numel(f)>=3 && all(diff(f)==f(2)-f(1)), ...
        'Noise estimation requires at least three equally spaced pilots.');
    xp = pilots(f,n);
    assert(all(abs(abs(xp)-1)<1e-10), 'This example expects unit-power pilots.');
    Nr = size(Y,3);
    hp = zeros(numel(f),Nr);
    Hls = zeros(size(Y));
    for r = 1:Nr
        % Eq. (2): unit-power pilots give H_LS = Y*conj(X) = Y/X.
        hp(:,r) = Y(f,n,r).*conj(xp);
        % Linear interpolation across frequency; extrapolate the edge bin.
        hFrequency = interp1(f, hp(:,r), (1:size(Y,1)).', 'linear', 'extrap');
        % Only one time observation: hold the estimate constant across time.
        Hls(:,:,r) = repmat(hFrequency, 1, size(Y,2));
    end

    % A raw LS residual Y-H_LS*X is zero, so it cannot estimate noise.
    % Instead, cancel a locally linear channel with a second difference:
    % d(k) = hp(k-1) - 2*hp(k) + hp(k+1).
    % Independent pilot noise gives E|d|^2 = (1+4+1)*sigma^2 = 6*sigma^2.
    % Average over pilot triples and antennas, assuming equal noise power.
    % This is our estimator choice; the paper does not specify this formula.
    % Channel curvature and ICI can bias it; it is not exact noise recovery.
    d = diff(hp,2,1);
    sigma2Hat = mean(abs(d(:)).^2)/6;
    details.Symbol = n;
    details.Subcarriers = f;
    details.RawChannelEstimates = hp;
end

function Xhat = equalizeSIMO(Y, H, sigma2Hat)
    % Eq. (3), specialized to ONE transmitted stream:
    % Xhat = sum_r conj(H_r)*Y_r / (sum_r |H_r|^2 + sigma^2).
    % sum(...,3) combines the receive antennas and leaves a frequency-time grid.
    gain = sum(abs(H).^2,3);
    Xhat = sum(conj(H).*Y,3) ./ max(gain + sigma2Hat,eps);
    % The LMMSE output is biased: E[Xhat|x] = g*x, g = gain/(gain+sigma^2) < 1.
    % Signs (QPSK) do not care, but 16QAM amplitude decisions would be pulled
    % toward the inner points. Remove g (unbiased LMMSE) before demapping.
    % For one stream this equals MRC, so sigma2Hat no longer changes hard
    % decisions; it still matters for soft reliability.
    Xhat = Xhat ./ max(gain ./ (gain + sigma2Hat),eps);
end

function bits = demapHard(symbols, modulation)
    % 5G mapping (TS 38.211 5.1.3/5.1.4). Negative component -> bit 1.
    % QPSK:  b1 real sign, b2 imaginary sign.
    % 16QAM: b1,b2 as QPSK (quadrant); b3,b4 = 1 for the OUTER level
    %        (|component| = 3/sqrt(10), beyond the 2/sqrt(10) boundary).
    q = 2*(1+strcmp(modulation,'16QAM'));
    bits = zeros(q*numel(symbols),1);
    bits(1:q:end) = real(symbols)<0;
    bits(2:q:end) = imag(symbols)<0;
    if q == 4
        bits(3:q:end) = abs(real(symbols))>2/sqrt(10);
        bits(4:q:end) = abs(imag(symbols))>2/sqrt(10);
    end
end

function checkReceiver(modulation)
    % Check every constellation label and a noiseless two-antenna link
    % independently of the saved data; also verify the noise estimator.
    q = 2*(1+strcmp(modulation,'16QAM'));
    b = reshape(dec2bin(0:2^q-1,q).'=='1',[],1);
    x = nrSymbolModulate(double(b),modulation);
    h = repmat(reshape([1+1i, -0.3+0.7i],1,1,2),numel(x),1,1);
    recovered = equalizeSIMO(h.*x,h,0.1);
    assert(isequal(demapHard(recovered,modulation),double(b)), ...
        '%s/SIMO check failed.',modulation);
    f = (1:72).';
    mask = false(72,14); mask(1:2:end,3) = true;
    pilots = double(mask);
    linearH = repmat(1+0.01*f+1i*0.02*f,1,14,2);
    [estimate,nvar] = estimateFromPilots(linearH.*pilots,pilots,mask);
    assert(max(abs(estimate(:)-linearH(:)))<1e-12 && nvar<1e-24);
    % Many independent receive branches provide a stable noise-power check.
    stream = RandStream('mt19937ar','Seed',123);
    y = sqrt(0.04/2)*(randn(stream,72,14,1000)+1i*randn(stream,72,14,1000));
    [~,nvar] = estimateFromPilots(y,pilots,mask);
    assert(abs(nvar/0.04-1)<0.05, 'Noise-power normalization check failed.');
    fprintf('Receiver checks passed.\n');
end

function plotReceiver(H, Hls, knownSymbols, lsSymbols, bers, errors, numBits, modulation)
    figure('Name','True and LS channel');
    Nr = size(H,3);
    magnitudeMax = max([abs(H(:));abs(Hls(:));eps]);
    errorMax = max([abs(H(:)-Hls(:));eps]);
    for r = 1:Nr
        grids = {abs(H(:,:,r)), abs(Hls(:,:,r)), abs(H(:,:,r)-Hls(:,:,r))};
        labels = {'True |H|','LS |H|','Absolute channel error'};
        for col = 1:3
            subplot(Nr,3,3*(r-1)+col);
            imagesc(grids{col}); axis xy;
            xlabel('OFDM symbol'); ylabel('Subcarrier'); colorbar;
            title(sprintf('%s, Rx %d',labels{col},r));
            if col<3, clim([0 magnitudeMax]); else, clim([0 errorMax]); end
        end
    end
    figure('Name','Receiver comparison');
    symbols = {knownSymbols,lsSymbols};
    names = {'Known H','LS H: one pilot symbol'};
    q = 2*(1+strcmp(modulation,'16QAM'));
    ideal = nrSymbolModulate(double(reshape(dec2bin(0:2^q-1,q).'=='1',[],1)),modulation);
    limit = 1.1*max([abs(real([knownSymbols;lsSymbols])); ...
        abs(imag([knownSymbols;lsSymbols]));1]);
    for k = 1:2
        subplot(1,3,k);
        plot(real(symbols{k}),imag(symbols{k}),'.'); hold on;
        plot(real(ideal),imag(ideal),'rx','LineWidth',2,'MarkerSize',10);
        axis equal; xlim([-limit limit]); ylim([-limit limit]); grid on;
        xlabel('In-phase'); ylabel('Quadrature'); title(names{k});
        legend('Equalized data',['Ideal ' modulation],'Location','best');
    end
    subplot(1,3,3);
    bar(1:2,bers); xticks(1:2); xticklabels({'Known H','LS H'});
    ylabel('Uncoded BER'); title(sprintf('%d data bits',numBits)); grid on;
    ylim([0 max(1/numBits,1.3*max(bers))]);
    for k = 1:2
        text(k,bers(k),sprintf(' %d errors; BER=%.3g',errors(k),bers(k)), ...
            'HorizontalAlignment','center','VerticalAlignment','bottom');
    end
end
