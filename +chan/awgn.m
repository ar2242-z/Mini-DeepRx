function [noise, noiseVariance] = awgn(numSamples, cfg, stream)
%AWGN Receiver thermal noise: white circular Gaussian time samples.
%   [noise, noiseVariance] = chan.awgn(numSamples, cfg, stream) returns
%   numSamples x cfg.NumReceiveAntennas noise samples, independent per
%   antenna. SNR is Es/N0 per occupied resource element per antenna,
%   averaged over channel realizations (unit symbol energy, unit average
%   channel power): N0 = 10^(-SNRdB/10). The receiver FFT is unscaled, so
%   variance N0/Nfft per time sample gives N0 on every FFT bin.
%   cfg.SNRdB = Inf returns zeros. noiseVariance is N0 (per FFT bin).
noiseVariance = 10^(-cfg.SNRdB/10);
dims = [numSamples cfg.NumReceiveAntennas];
if noiseVariance == 0
    noise = zeros(dims);
    return
end
noise = sqrt(noiseVariance/cfg.Nfft/2) * ...
    (randn(stream, dims) + 1i*randn(stream, dims));
end
