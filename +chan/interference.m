function [interference, out] = interference(numSamples, carrier, cfg)
%INTERFERENCE One co-channel interferer: another uplink user in a nearby cell.
%   [interference, out] = chan.interference(numSamples, carrier, cfg)
%   returns the interferer's received waveform (numSamples x antennas),
%   aligned sample for sample with chan.fading's output for the wanted user.
%   The interferer is a full transmitter of its own (tx.transmit with cell
%   ID 1: own bits, own DMRS scrambling, same PA model) through its own
%   fading channel with the same model, delay spread and Doppler.
%   Power: both users have unit symbol energy and unit average channel
%   power, so scaling by 10^(-SIRdB/20) sets the average interference power
%   per occupied resource element to 10^(-SIRdB/10) of the wanted signal.
%   cfg.SIRdB = Inf returns zeros. The interferer's random draws come from
%   their own stream (seeded by cfg.Seed), so they never change the wanted
%   user's bits, fading or noise.
%   out: Seed (interferer's slot seed), Tx (its tx.transmit output), H.
numAntennas = cfg.NumReceiveAntennas;
out = struct('Seed',[],'Tx',[],'H',[]);
if ~isfinite(cfg.SIRdB)
    interference = zeros(numSamples, numAntennas);
    return
end
stream = RandStream('mrg32k3a', 'Seed', cfg.Seed);
icfg = cfg;
icfg.Seed = randi(stream, [0 2^32-1]);
while icfg.Seed == cfg.Seed
    icfg.Seed = randi(stream, [0 2^32-1]);
end
[waveform, out.Tx] = tx.transmit(icfg, stream, 1);
[received, fade] = chan.fading(waveform, out.Tx.Carrier, icfg, icfg.Seed);
assert(size(received,1) == numSamples, ...
    'Interferer has %d samples; the wanted signal has %d.', size(received,1), numSamples);
interference = sqrt(10^(-cfg.SIRdB/10)) * received;
out.Seed = icfg.Seed;
out.H = fade.H;
end
