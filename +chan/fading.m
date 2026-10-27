function [rxWaveform, out] = fading(waveform, carrier, cfg, seed)
%FADING Multipath fading channel (3GPP TR 38.901 TDL or CDL) in time.
%   [rxWaveform, out] = chan.fading(waveform, carrier, cfg, seed) passes the
%   transmit waveform (one column, one antenna) through a time-varying
%   channel to cfg.NumReceiveAntennas receive antennas. Physically: the signal
%   arrives over several delayed, attenuated paths, each with its own Doppler
%   (cfg.MaximumDopplerShift), so the channel is frequency selective and
%   changes over the slot. Zeros are appended so the delayed end of the slot
%   is not cut off; rxWaveform therefore has out.Padding extra samples.
%   seed fixes the fading realization. Average channel power is one.
%   out holds the truth used for evaluation and the known-H baseline:
%     H              subcarrier x symbol x antenna channel response, using
%                    the perfect timing below
%     TimingOffset   samples from the start of rxWaveform to the start of
%                    the slot (strongest-path timing, a genie value)
%     SampleRate, Padding, PathDelays, AveragePathGainsDB
ofdm = nrOFDMInfo(carrier, 'Nfft', cfg.Nfft);
channel = makeChannel(cfg, ofdm.SampleRate, seed);
channelInfo = info(channel);
% Multipath and filtering delay the signal; padding flushes the last symbol.
padding = ceil(max(channelInfo.PathDelays)*ofdm.SampleRate) ...
    + channelInfo.ChannelFilterDelay;
[rxWaveform, pathGains, sampleTimes] = channel([waveform; zeros(padding,1)]);
pathFilters = getPathFilters(channel);
release(channel);

offset = nrPerfectTimingEstimate(pathGains, pathFilters);
numSymbols = carrier.SymbolsPerSlot;
H = nrPerfectChannelEstimate(carrier, pathGains, pathFilters, ...
    offset, sampleTimes, 'Nfft', cfg.Nfft);
out.H = H(:,1:numSymbols,:,1); % This slot; the only transmit antenna.
out.TimingOffset = offset;
out.SampleRate = ofdm.SampleRate;
out.Padding = padding;
out.PathDelays = channelInfo.PathDelays;
out.AveragePathGainsDB = channelInfo.AveragePathGains;
end

function channel = makeChannel(cfg, sampleRate, seed)
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
        % Isotropic, single-polarized, VERTICAL elements at both ends keep the
        % average channel power at one, as for TDL, so SNRdB means the same
        % for all ten models. The toolbox default slants a single element at
        % 45 degrees, which captures only about half the power of CDL's
        % cross-polarized rays (measured 0.50-0.60).
        channel.TransmitAntennaArray.Size = [1 1 1 1 1];
        channel.TransmitAntennaArray.Element = 'isotropic';
        channel.TransmitAntennaArray.PolarizationAngles = [0 90];
        channel.ReceiveAntennaArray.Size = [1 cfg.NumReceiveAntennas 1 1 1];
        channel.ReceiveAntennaArray.Element = 'isotropic';
        channel.ReceiveAntennaArray.PolarizationAngles = [0 90];
    end
    channel.DelayProfile = cfg.ChannelModel;
    channel.DelaySpread = cfg.DelaySpread;
    channel.MaximumDopplerShift = cfg.MaximumDopplerShift;
    channel.NormalizePathGains = true;
    channel.NormalizeChannelOutputs = false;
    channel.SampleRate = sampleRate;
    channel.RandomStream = 'mt19937ar with seed';
    channel.Seed = seed;
end
