function [X, H, Y, meta] = mini_simo_channel(cfg)
%% Mini SIMO experiment: one transmitter and two receivers
% Requires MATLAB and 5G Toolbox. Open this file and click Run.
% Or call it with settings from simo_config:
%   cfg = simo_config('SNRdB', 5, 'MaximumDopplerShift', 200);
%   [X,H,Y,meta] = mini_simo_channel(cfg);
% X(f,n)   : transmitted complex symbol (data, pilot, or zero).
% H(f,n,r) : true channel response, NOT a pilot-based estimate.
% Y(f,n,r) : received complex symbol, including noise (and interference
%            when cfg.SIRdB is finite).
% meta.TimeDomainY(t,n,r): the same received signal BEFORE the FFT, with
%            CP, one column per OFDM symbol (552 x 14 x 2 by default).
% n = OFDM symbol (time), f = active subcarrier, r = receive antenna.
% All grids use toolbox order: (frequency,time,antenna).
% X has one transmit antenna; MATLAB omits its trailing singleton dimension.
% This function generates one slot; it does not train DeepRx yet.
% Helper functions are below the main function in this same file.

%% 1. Use default settings unless the caller supplies a configuration
if nargin == 0
    cfg = simo_config('DMRSAdditionalPosition', 0);
else
    % Validate edits made directly to cfg fields as well as named overrides.
    names = fieldnames(cfg);
    values = struct2cell(cfg);
    options = [names.'; values.'];
    cfg = simo_config(options{:});
end

assert(exist('nrTDLChannel','class') == 8 && exist('nrCDLChannel','class') == 8, ...
    'This example requires MATLAB 5G Toolbox.');

%% 2. Generate the transmitted grid and propagate it through the channel
[X, H, Y, meta] = generateSlot(cfg);
% meta contains the bits, data/pilot masks, noise, and noiseless reception.
% A mask is a logical array: true marks a resource element of that type.
if cfg.Verbose
fprintf('X: %s, H: %s, Y: %s\n', ...
    mat2str(size(X)), mat2str(size(H)), mat2str(size(Y)));
fprintf('Noise variance per complex resource element: %.4g\n', ...
    meta.NoiseVariance);
fprintf('SIR: %g dB\n', meta.SIRdB);
if ~strcmp(meta.PAModel,'none')
    fprintf('PA: %s, %g dB back-off, gain %.4g\n', ...
        meta.PAModel, meta.PABackoffdB, abs(meta.PAGain));
end
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
    outputFile = fullfile(fileparts(mfilename('fullpath')), 'mini_simo_channel_data.mat');
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

    [X, dataMask, pilotMask, bits, dataIndices, bitsPerSymbol] = ...
        makeGrid(carrier, cfg, stream);
    numSymbols = size(X,2); % Columns are OFDM symbols (time).

    channel = makeChannel(cfg, ofdm.SampleRate);

    % Turn subcarrier symbols into time samples using IFFT and cyclic prefix.
    % Disable transmit windowing to keep this introductory example simple.
    waveform = nrOFDMModulate(carrier, X, 'Windowing', 0, 'Nfft', cfg.Nfft);
    % Optional PA nonlinearity. H below comes from the path gains, so it stays
    % the linear channel; PA distortion appears in Y as an extra error term.
    [waveform, paGain] = applyPA(waveform, cfg);
    channelInfo = info(channel);
    % Multipath/filtering delays the signal. Extra zeros flush the last symbol.
    padding = ceil(max(channelInfo.PathDelays)*ofdm.SampleRate) ...
        + channelInfo.ChannelFilterDelay;
    [rxWaveform, pathGains, sampleTimes] = ...
        channel([waveform; zeros(padding,1)]);
    pathFilters = getPathFilters(channel);
    offset = nrPerfectTimingEstimate(pathGains, pathFilters);
    % Keep exactly this slot's samples: noise and interference are added to
    % these time samples, so Y and the time grid below share one realization.
    cpLengths = double(ofdm.CyclicPrefixLengths(1:numSymbols));
    slotLength = sum(cpLengths) + numSymbols*cfg.Nfft;
    rxWaveform = rxWaveform(offset+(1:slotLength),:);
    rxGrid = nrOFDMDemodulate(carrier, rxWaveform, 'Nfft', cfg.Nfft);

    % Derive ground-truth H from the simulated paths using the SAME timing
    % offset as Y. No division by X and no noisy pilot estimation is used.
    hGrid = nrPerfectChannelEstimate(carrier, pathGains, pathFilters, ...
        offset, sampleTimes, 'Nfft', cfg.Nfft);
    % Keep this slot and the only transmit antenna (fourth dimension of H).
    H = hGrid(:,1:numSymbols,:,1);
    noiselessY = rxGrid(:,1:numSymbols,:);

    % Optional co-channel interferer: another PUSCH with its own bits, DMRS
    % scrambling (cell ID 1), and fading. It shares the channel model, delay
    % spread, and Doppler, and the receiver keeps the wanted user's timing.
    % Its private stream leaves the wanted user's bits, fading, and noise
    % identical to a slot without interference.
    interferenceTime = zeros(size(rxWaveform));
    if isfinite(cfg.SIRdB)
        interferenceStream = RandStream('mrg32k3a', 'Seed', cfg.Seed);
        interfererCfg = cfg;
        interfererCfg.Seed = randi(interferenceStream, [0 2^32-1]);
        while interfererCfg.Seed == cfg.Seed
            interfererCfg.Seed = randi(interferenceStream, [0 2^32-1]);
        end
        interfererCarrier = carrier;
        interfererCarrier.NCellID = 1;
        interfererX = makeGrid(interfererCarrier, interfererCfg, interferenceStream);
        interfererChannel = makeChannel(interfererCfg, ofdm.SampleRate);
        interfererWaveform = nrOFDMModulate(interfererCarrier, interfererX, ...
            'Windowing', 0, 'Nfft', cfg.Nfft);
        % The interferer is another UE with its own PA of the same model.
        interfererWaveform = applyPA(interfererWaveform, cfg);
        interfererRx = interfererChannel([interfererWaveform; zeros(padding,1)]);
        release(interfererChannel);
        % Both users have unit symbol power and normalized channels, so this
        % scaling sets the average interference power per occupied RE.
        interferenceTime = sqrt(10^(-cfg.SIRdB/10)) * ...
            interfererRx(offset+(1:slotLength),:);
    end

    % 5G QPSK/16QAM have unit AVERAGE symbol power (16QAM scaled by 1/sqrt(10))
    % and the average channel power is normalized.
    % SNR is Es/N0 per occupied resource element, per receive branch, averaged
    % over channel realizations. It is NOT the measured SNR of this one slot,
    % nor the full-band SNR convention in the paper.
    % White circular Gaussian noise is added to the TIME samples. The
    % demodulator is an unscaled FFT, so variance noiseVariance/Nfft per
    % sample gives noiseVariance on every FFT bin, as before.
    % SIR uses the same per-RE reference as SNR.
    noiseVariance = 10^(-cfg.SNRdB/10);
    noiseTime = sqrt(noiseVariance/cfg.Nfft/2) * ...
        (randn(stream,size(rxWaveform)) + 1i*randn(stream,size(rxWaveform)));
    receivedTime = rxWaveform + interferenceTime + noiseTime;
    demodulate = @(w) nrOFDMDemodulate(carrier, w, 'Nfft', cfg.Nfft);
    Y = demodulate(receivedTime);
    noise = demodulate(noiseTime);               % Linear, so Y is the sum of
    interference = demodulate(interferenceTime); % noiselessY, interference, noise.
    [Yt, timeGrid] = makeTimeGrid(receivedTime, cpLengths, cfg.Nfft, size(X,1));

    meta.DataMask = dataMask;
    meta.PilotMask = pilotMask;
    meta.Bits = bits;
    % With Q = meta.BitsPerSymbol, bits(Q*(k-1)+(1:Q)) label X(meta.DataIndices(k)).
    meta.DataIndices = dataIndices;
    meta.Modulation = cfg.Modulation;
    meta.ChannelModel = cfg.ChannelModel;
    meta.PilotSymbols = find(any(pilotMask,1)); % MATLAB columns with pilots.
    meta.BitsPerSymbol = bitsPerSymbol;
    meta.NoiseVariance = noiseVariance;
    meta.Noise = noise;
    meta.SIRdB = cfg.SIRdB;
    meta.Interference = interference; % Zero when SIRdB is Inf.
    meta.NoiselessY = noiselessY;     % Wanted user only: no interference.
    % Received samples BEFORE the FFT, one column per OFDM symbol, aligned
    % so the FFT window is the same rows in every column (see makeTimeGrid).
    meta.TimeDomainY = Yt;
    meta.TimeGrid = timeGrid;
    meta.PAModel = cfg.PAModel;
    meta.PABackoffdB = cfg.PABackoffdB;
    meta.PAGain = paGain;             % Bussgang gain; 1 when PAModel is 'none'.
    meta.TimingOffsetSamples = offset;
    meta.SampleRate = ofdm.SampleRate;
    meta.PathDelaysSeconds = channelInfo.PathDelays;
    meta.AveragePathGainsDB = channelInfo.AveragePathGains;
    meta.DiagonalModelError = norm(noiselessY(:) - ...
        reshape(H.*X,[],1)) / norm(noiselessY(:));
    release(channel);
end

function channel = makeChannel(cfg, sampleRate)
    % TDL: tapped delay line, antennas set by count and correlation.
    % CDL: clustered paths with angles, antennas set by array geometry.
    if startsWith(cfg.ChannelModel,'TDL')
        channel = nrTDLChannel;
        channel.NumTransmitAntennas = 1;
        channel.NumReceiveAntennas = cfg.NumReceiveAntennas;
        channel.MIMOCorrelation = 'Low';
    else
        channel = nrCDLChannel;
        channel.CarrierFrequency = cfg.CarrierFrequency;
        % Array size is [rows columns polarizations panelRows panelColumns].
        % Isotropic elements keep average channel power at one, as for TDL,
        % so SNRdB keeps the same meaning across all ten models.
        channel.TransmitAntennaArray.Size = [1 1 1 1 1];
        channel.TransmitAntennaArray.Element = 'isotropic';
        channel.ReceiveAntennaArray.Size = [1 cfg.NumReceiveAntennas 1 1 1];
        channel.ReceiveAntennaArray.Element = 'isotropic';
    end
    channel.DelayProfile = cfg.ChannelModel;
    channel.DelaySpread = cfg.DelaySpread;
    channel.MaximumDopplerShift = cfg.MaximumDopplerShift;
    channel.NormalizePathGains = true;
    channel.NormalizeChannelOutputs = false;
    channel.SampleRate = sampleRate;
    channel.RandomStream = 'mt19937ar with seed';
    channel.Seed = cfg.Seed;
end

function [y, gain] = applyPA(x, cfg)
    % Memoryless PA on the time waveform. The input is scaled so its RMS sits
    % PABackoffdB below the saturation amplitude (1). The output is scaled back
    % to the input RMS, so SNRdB keeps its meaning and only distortion is added.
    gain = 1;
    if strcmp(cfg.PAModel,'none')
        y = x;
        return
    end
    rmsIn = sqrt(mean(abs(x).^2));
    u = x/rmsIn * 10^(-cfg.PABackoffdB/20);
    switch cfg.PAModel
        case 'rapp'
            p = cfg.RappSmoothness;
            v = u ./ (1 + abs(u).^(2*p)).^(1/(2*p));
        case 'polynomial'
            % Hold |u| at saturation so a falling AM/AM curve cannot fold back.
            u = min(abs(u),1) .* exp(1i*angle(u));
            a = cfg.PACoefficients;
            v = zeros(size(u));
            for k = 1:numel(a)
                v = v + a(k)*u.*abs(u).^(2*k-2);
            end
    end
    y = v * (rmsIn/sqrt(mean(abs(v).^2)));
    % Bussgang gain: y = gain*x + distortion uncorrelated with x.
    gain = (x'*y)/(x'*x);
end

function [Yt, info] = makeTimeGrid(waveform, cpLengths, Nfft, numSubcarriers)
    % One column per OFDM symbol: its cyclic prefix and Nfft samples, zero
    % padded so the FFT window lands on the SAME rows (info.FFTRows) in every
    % column. Short-CP symbols get the extra zeros. nrOFDMDemodulate starts
    % its FFT window fix(0.5*cp) samples into the CP (CyclicPrefixFraction
    % default) and removes the resulting phase ramp, so with
    %   Fx = fft(Yt(info.FFTRows,:,:));
    %   Y  = Fx(info.ActiveBins,:,:) .* info.PhaseCorrection;
    % reproduces Y exactly. The time CNN relies on this mapping.
    numSymbols = numel(cpLengths);
    skip = fix(0.5*cpLengths);   % CP samples before the FFT window.
    after = cpLengths - skip;    % CP samples moved to the end by the shift.
    before = max(skip);
    numRows = before + Nfft + max(after);
    starts = cumsum([0 cpLengths(1:end-1)+Nfft]);
    Yt = zeros(numRows, numSymbols, size(waveform,2), 'like', waveform);
    for n = 1:numSymbols
        rows = before - skip(n) + (1:cpLengths(n)+Nfft);
        Yt(rows,n,:) = waveform(starts(n)+(1:cpLengths(n)+Nfft),:);
    end
    k = (0:numSubcarriers-1).' - numSubcarriers/2; % Subcarrier offset from DC.
    info.Nfft = Nfft;
    info.CyclicPrefixLengths = cpLengths;
    info.FFTRows = before + (1:Nfft);
    info.ActiveBins = mod(k, Nfft) + 1;
    info.PhaseCorrection = exp(2i*pi*k*after/Nfft);
end

function [grid, dataMask, pilotMask, bits, dataIndices, bitsPerSymbol] = ...
        makeGrid(carrier, cfg, stream)
    % Use standard uplink pilot positions and sequences from 5G Toolbox.
    pusch = nrPUSCHConfig;
    pusch.PRBSet = 0:carrier.NSizeGrid-1; % Toolbox PRB numbers start at zero.
    pusch.SymbolAllocation = [0 14];    % Start at symbol 0; use 14 symbols.
    pusch.MappingType = 'A';
    pusch.NumLayers = 1;
    modulation = cfg.Modulation;
    pusch.Modulation = modulation;
    pusch.TransformPrecoding = false;
    pusch.EnablePTRS = false;
    pusch.DMRS.DMRSTypeAPosition = cfg.DMRSTypeAPosition; % 2 means MATLAB column 3.
    pusch.DMRS.DMRSLength = cfg.DMRSLength;
    pusch.DMRS.DMRSAdditionalPosition = cfg.DMRSAdditionalPosition; % Extra pilot symbols.
    pusch.DMRS.DMRSConfigurationType = cfg.DMRSConfigurationType;
    pusch.DMRS.NumCDMGroupsWithoutData = cfg.NumCDMGroupsWithoutData;
    % Unused CDM groups stay empty. Their 3 dB DMRS power boost is not applied:
    % pilots keep unit magnitude, which the LS receiver relies on.
    dataIndices = nrPUSCHIndices(carrier, pusch);
    pilotIndices = nrPUSCHDMRSIndices(carrier, pusch);
    pilots = nrPUSCHDMRS(carrier, pusch);
    bitsPerSymbol = 2*(1+strcmp(modulation,'16QAM')); % QPSK: 2, 16QAM: 4.
    bits = randi(stream, [0 1], bitsPerSymbol*numel(dataIndices), 1);
    symbols = nrSymbolModulate(bits, modulation); % Unit average power.
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
    assert(numel(meta.Bits) == meta.BitsPerSymbol*numel(meta.DataIndices));
    expectedSymbols = nrSymbolModulate(meta.Bits, meta.Modulation);
    assert(max(abs(X(meta.DataIndices)-expectedSymbols)) < 1e-12);
    % The time grid must reproduce Y through the documented FFT mapping.
    g = meta.TimeGrid;
    Fx = fft(meta.TimeDomainY(g.FFTRows,:,:));
    timeError = norm(reshape(Fx(g.ActiveBins,:,:).*g.PhaseCorrection - Y,[],1)) / norm(Y(:));
    assert(timeError < 1e-9, 'Time grid does not reproduce Y.');
    % Noise must keep variance 10^(-SNRdB/10) per resource element after FFT.
    if isfinite(cfg.SNRdB)
        noiseRatio = mean(abs(meta.Noise(:)).^2) / meta.NoiseVariance;
        assert(abs(noiseRatio-1) < 0.1, 'Noise scaling check failed.');
    end

    % A separate static, noiseless run checks FFT scaling and timing alignment.
    staticCfg = cfg;
    staticCfg.MaximumDopplerShift = 0;
    staticCfg.SNRdB = Inf;
    staticCfg.SIRdB = Inf;
    staticCfg.PAModel = 'none'; % PA distortion is real, not a scaling bug.
    % Long-delay models (A, D, E at 300 ns) put paths near or beyond the
    % cyclic prefix; that real inter-symbol interference is not a scaling bug.
    staticCfg.DelaySpread = min(cfg.DelaySpread, 30e-9);
    [~,~,~,staticMeta] = generateSlot(staticCfg);
    assert(staticMeta.DiagonalModelError < 1e-3, ...
        'Static channel check failed: check timing, CP length, and scaling.');
    fprintf('Checks passed. Static/noiseless relative error: %.3g\n', ...
        staticMeta.DiagonalModelError);
end

function plotExample(X, H, meta)
    figure('Name',['Mini SIMO ' meta.ChannelModel]);
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
