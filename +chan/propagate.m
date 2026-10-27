function [rxWaveform, out] = propagate(waveform, carrier, cfg, noiseStream)
%PROPAGATE The whole channel: transmit waveform to receive-antenna samples.
%   [rxWaveform, out] = chan.propagate(waveform, carrier, cfg, noiseStream)
%     fading (seed cfg.Seed) -> + interference (cfg.SIRdB) -> + noise (cfg.SNRdB)
%   rxWaveform is samples x receive antennas, including chan.fading's
%   padding; finding the slot inside it is the receiver's job.
%   Noise comes from noiseStream, so it is independent of the bits.
%   out holds the parts of the sum and the truth (Rx must not use them):
%     Noiseless, Interference, Noise   the three time-domain parts
%     H, TimingOffset, SampleRate, Padding, PathDelays, AveragePathGainsDB
%                                      from chan.fading
%     NoiseVariance                    N0 per resource element
%     Interferer                       from chan.interference
[noiseless, out] = chan.fading(waveform, carrier, cfg, cfg.Seed);
numSamples = size(noiseless,1);
[interference, out.Interferer] = chan.interference(numSamples, carrier, cfg);
[noise, out.NoiseVariance] = chan.awgn(numSamples, cfg, noiseStream);
rxWaveform = noiseless + interference + noise;
out.Noiseless = noiseless;
out.Interference = interference;
out.Noise = noise;
end
