function [rxWaveform, truth, trace] = run_link(cfg)
%RUN_LINK Transmitter and channel for one slot: what the receiver gets.
%   [rxWaveform, truth, trace] = run_link(cfg)
%   rxWaveform  samples x receive antennas; the only receiver input
%               (together with rx.context(cfg)).
%   truth       for evaluation and genie baselines only:
%               InfoBits, CodedBits, Code, Grid, H, TimingOffset, NoiseVariance
%   trace       tx.transmit and chan.propagate outputs (all checkpoints)
% Random streams, all derived from cfg.Seed and independent of each other:
%   bits         mt19937ar, seed cfg.Seed
%   fading       the channel's own generator, seed cfg.Seed
%   interferer   mrg32k3a, seed cfg.Seed, substream 1 (chan.interference)
%   noise        mrg32k3a, seed cfg.Seed, substream 2
% So changing the coding (number of bits) never changes fading or noise.
bitStream = RandStream('mt19937ar', 'Seed', cfg.Seed);
noiseStream = RandStream('mrg32k3a', 'Seed', cfg.Seed);
noiseStream.Substream = 2;

[waveform, txOut] = tx.transmit(cfg, bitStream);
[rxWaveform, chOut] = chan.propagate(waveform, txOut.Carrier, cfg, noiseStream);

truth.InfoBits = txOut.InfoBits;
truth.CodedBits = txOut.CodedBits;
truth.Code = txOut.Code;
truth.Grid = txOut.Grid;
truth.H = chOut.H;
truth.TimingOffset = chOut.TimingOffset;
truth.NoiseVariance = chOut.NoiseVariance;
trace.Tx = txOut;
trace.Channel = chOut;
end
