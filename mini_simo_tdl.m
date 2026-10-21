function [X, H, Y, meta] = mini_simo_tdl(cfg)
%% Mini SIMO experiment: one transmitter and two receivers
% Requires MATLAB and 5G Toolbox. Open this file and click Run.
% Or call it with settings from simo_config:
%   cfg = simo_config('SNRdB', 5, 'MaximumDopplerShift', 200);
%   [X,H,Y,meta] = mini_simo_tdl(cfg);
% X(f,n)   : transmitted complex symbol (data, pilot, or zero).
% H(f,n,r) : true channel response, NOT a pilot-based estimate.
% Y(f,n,r) : received complex symbol, including noise.
% n = OFDM symbol (time), f = active subcarrier, r = receive antenna.
% All grids use toolbox order: (frequency,time,antenna).
% X has one transmit antenna; MATLAB omits its trailing singleton dimension.
% This function generates one slot; it does not train DeepRx yet.
% Helper functions are below the main function in this same file.

%% 1. Use default settings unless the caller supplies a configuration
if nargin == 0
    cfg = simo_config();
else
    % Validate edits made directly to cfg fields as well as named overrides.
    names = fieldnames(cfg);
    values = struct2cell(cfg);
    options = [names.'; values.'];
    cfg = simo_config(options{:});
end

assert(exist('nrTDLChannel','class') == 8, ...
    'This example requires MATLAB 5G Toolbox.');

%% 2. Generate the transmitted grid and propagate it through TDL-C
[X, H, Y, meta] = generateSlot(cfg);
% meta contains the bits, data/pilot masks, noise, and noiseless reception.
% A mask is a logical array: true marks a resource element of that type.
if cfg.Verbose
fprintf('X: %s, H: %s, Y: %s\n', ...
    mat2str(size(X)), mat2str(size(H)), mat2str(size(Y)));
fprintf('Noise variance per complex resource element: %.4g\n', ...
    meta.NoiseVariance);
fprintf('Noiseless Y versus H.*X relative error: %.4g\n', ...
    meta.DiagonalModelError);
end
% At nonzero Doppler this error need not vanish: the channel also changes
% WITHIN each OFDM symbol, causing intercarrier interference (ICI).

%% 3. Verify the simulation and save the useful arrays
if cfg.RunChecks
    checkSimulation(X, H, Y, meta, cfg);
end
if cfg.SaveData
    outputFile = fullfile(fileparts(mfilename('fullpath')), 'mini_simo_tdl_data.mat');
    save(outputFile, 'X', 'H', 'Y', 'cfg', 'meta');
    if cfg.Verbose, fprintf('Saved: %s\n', outputFile); end
end

%% 4. Display the pilot locations and the two channel responses
if cfg.ShowPlots
    plotExample(X, H, meta);
end
end

%% Local functions: implementation details below this line
function [X, H, Y, meta] = generateSlot(cfg)
    % A private random stream avoids changing MATLAB's global random state.
    stream = RandStream('mt19937ar', 'Seed', cfg.Seed);
    carrier = nrCarrierConfig;
    carrier.NSizeGrid = cfg.NumResourceBlocks;
    carrier.SubcarrierSpacing = cfg.SubcarrierSpacing;
    carrier.CyclicPrefix = 'normal';
    carrier.NSlot = 0;
    carrier.NCellID = 0;
    ofdm = nrOFDMInfo(carrier, 'Nfft', cfg.Nfft);

    [X, dataMask, pilotMask, bits, dataIndices] = makeGrid(carrier, stream);
    numSymbols = size(X,2); % Columns are OFDM symbols (time).

    channel = nrTDLChannel;
    channel.DelayProfile = 'TDL-C';
    channel.DelaySpread = cfg.DelaySpread;
    channel.MaximumDopplerShift = cfg.MaximumDopplerShift;
    channel.NumTransmitAntennas = 1;
    channel.NumReceiveAntennas = cfg.NumReceiveAntennas;
    channel.MIMOCorrelation = 'Low';
    channel.NormalizePathGains = true;
    channel.NormalizeChannelOutputs = false;
    channel.SampleRate = ofdm.SampleRate;
    channel.RandomStream = 'mt19937ar with seed';
    channel.Seed = cfg.Seed;

    % Turn subcarrier symbols into time samples using IFFT and cyclic prefix.
    % Disable transmit windowing to keep this introductory example simple.
    waveform = nrOFDMModulate(carrier, X, 'Windowing', 0, 'Nfft', cfg.Nfft);
    channelInfo = info(channel);
    % Multipath/filtering delays the signal. Extra zeros flush the last symbol.
    padding = ceil(max(channelInfo.PathDelays)*ofdm.SampleRate) ...
        + channelInfo.ChannelFilterDelay;
    [rxWaveform, pathGains, sampleTimes] = ...
        channel([waveform; zeros(padding,1)]);
    pathFilters = getPathFilters(channel);
    offset = nrPerfectTimingEstimate(pathGains, pathFilters);
    rxWaveform = rxWaveform(1+offset:end,:);
    rxGrid = nrOFDMDemodulate(carrier, rxWaveform, 'Nfft', cfg.Nfft);

    % Derive ground-truth H from the simulated paths using the SAME timing
    % offset as Y. No division by X and no noisy pilot estimation is used.
    hGrid = nrPerfectChannelEstimate(carrier, pathGains, pathFilters, ...
        offset, sampleTimes, 'Nfft', cfg.Nfft);
    % Keep this slot and the only transmit antenna (fourth dimension of H).
    H = hGrid(:,1:numSymbols,:,1);
    noiselessY = rxGrid(:,1:numSymbols,:);

    % QPSK has unit symbol power and the average channel power is normalized.
    % SNR is Es/N0 per occupied resource element, per receive branch, averaged
    % over channel realizations. It is NOT the measured SNR of this one slot,
    % nor the full-band SNR convention in the paper.
    % Add circular complex Gaussian noise after FFT: this is equivalent to
    % white receiver noise on these FFT bins and makes its variance explicit.
    noiseVariance = 10^(-cfg.SNRdB/10);
    noise = sqrt(noiseVariance/2) * ...
        (randn(stream,size(H)) + 1i*randn(stream,size(H)));
    Y = noiselessY + noise;

    meta.DataMask = dataMask;
    meta.PilotMask = pilotMask;
    meta.Bits = bits;
    % bits(2*k-1:2*k) label X(meta.DataIndices(k)).
    meta.DataIndices = dataIndices;
    meta.NoiseVariance = noiseVariance;
    meta.Noise = noise;
    meta.NoiselessY = noiselessY;
    meta.TimingOffsetSamples = offset;
    meta.SampleRate = ofdm.SampleRate;
    meta.PathDelaysSeconds = channelInfo.PathDelays;
    meta.AveragePathGainsDB = channelInfo.AveragePathGains;
    meta.DiagonalModelError = norm(noiselessY(:) - ...
        reshape(H.*X,[],1)) / norm(noiselessY(:));
    release(channel);
end

function [grid, dataMask, pilotMask, bits, dataIndices] = makeGrid(carrier, stream)
    % Use standard uplink pilot positions and sequences from 5G Toolbox.
    pusch = nrPUSCHConfig;
    pusch.PRBSet = 0:carrier.NSizeGrid-1; % Toolbox PRB numbers start at zero.
    pusch.SymbolAllocation = [0 14];    % Start at symbol 0; use 14 symbols.
    pusch.MappingType = 'A';
    pusch.NumLayers = 1;
    pusch.Modulation = 'QPSK';
    pusch.TransformPrecoding = false;
    pusch.EnablePTRS = false;
    pusch.DMRS.DMRSTypeAPosition = 2;    % Symbol 2 means MATLAB column 3.
    pusch.DMRS.DMRSLength = 1;
    pusch.DMRS.DMRSAdditionalPosition = 0; % One pilot-bearing symbol.
    pusch.DMRS.DMRSConfigurationType = 1;
    pusch.DMRS.NumCDMGroupsWithoutData = 1;
    dataIndices = nrPUSCHIndices(carrier, pusch);
    pilotIndices = nrPUSCHDMRSIndices(carrier, pusch);
    pilots = nrPUSCHDMRS(carrier, pusch);
    bits = randi(stream, [0 1], 2*numel(dataIndices), 1);
    symbols = nrSymbolModulate(bits, 'QPSK'); % Two bits per unit-power symbol.
    grid = nrResourceGrid(carrier, 1);
    grid(dataIndices) = symbols;
    grid(pilotIndices) = pilots;
    dataMask = false(size(grid));
    pilotMask = false(size(grid));
    dataMask(dataIndices) = true;
    pilotMask(pilotIndices) = true;
end

function checkSimulation(X, H, Y, meta, cfg)
    assert(isequal(size(X), [12*cfg.NumResourceBlocks 14]));
    assert(size(H,1)==size(X,1) && size(H,2)==size(X,2));
    assert(size(H,3)==cfg.NumReceiveAntennas && isequal(size(H),size(Y)));
    assert(all(isfinite(Y(:))) && all(isfinite(H(:))));
    assert(~any(meta.DataMask(:) & meta.PilotMask(:)));
    expectedSymbols = nrSymbolModulate(meta.Bits, 'QPSK');
    assert(max(abs(X(meta.DataIndices)-expectedSymbols)) < 1e-12);

    % A separate static, noiseless run checks FFT scaling and timing alignment.
    staticCfg = cfg;
    staticCfg.MaximumDopplerShift = 0;
    staticCfg.SNRdB = Inf;
    [~,~,~,staticMeta] = generateSlot(staticCfg);
    assert(staticMeta.DiagonalModelError < 1e-3, ...
        'Static channel check failed: check timing, CP length, and scaling.');
    fprintf('Checks passed. Static/noiseless relative error: %.3g\n', ...
        staticMeta.DiagonalModelError);
end

function plotExample(X, H, meta)
    figure('Name','Mini SIMO TDL-C');
    subplot(1,3,1);
    % imagesc places columns (OFDM symbols) on the horizontal axis.
    gridTypes = double(meta.DataMask)+2*double(meta.PilotMask);
    imagesc(1:size(X,2), 1:size(X,1), gridTypes);
    axis xy; xlabel('OFDM symbol'); ylabel('Subcarrier');
    title('Grid: 0=unused, 1=data, 2=pilot'); colorbar;
    for r = 1:min(size(H,3),2)
        subplot(1,3,r+1);
        imagesc(1:size(X,2), 1:size(X,1), abs(H(:,:,r)));
        axis xy; xlabel('OFDM symbol'); ylabel('Subcarrier');
        title(sprintf('|H|, receive antenna %d',r)); colorbar;
    end
end
