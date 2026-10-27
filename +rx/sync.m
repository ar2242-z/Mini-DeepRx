function [slot, offset, metric] = sync(rxWaveform, ctx, perfectOffset)
%SYNC Timing synchronization: find the slot inside the received samples.
%   [slot, offset] = rx.sync(rxWaveform, ctx) estimates where the slot
%   starts and returns its ctx.SlotLength samples (all antennas, CPs kept).
%   offset counts the samples before the slot start (0 = no delay).
%   ctx.TimingEstimation selects the method:
%     'dmrs'     correlate each antenna with the known pilot-only waveform
%                (ctx.PilotWaveform) and take the lag with the largest
%                power summed over antennas: the strongest path, which is
%                also how the channel's true timing is defined.
%     'perfect'  use perfectOffset, the channel's true timing (genie);
%                for reference runs only.
%   metric is the correlation power per lag (empty for 'perfect').
%   Not differentiable (argmax); neural receivers act after this block.
L = size(rxWaveform, 1);
maxLag = L - ctx.SlotLength;
assert(maxLag >= 0, 'Received waveform is shorter than one slot.');
metric = [];
switch ctx.TimingEstimation
    case 'perfect'
        assert(nargin == 3, 'Perfect timing needs the true offset.');
        offset = perfectOffset;
    case 'dmrs'
        % Cross-correlation r(d) = sum_t rx(t+d) conj(ref(t)) for d >= 0,
        % via FFT; the length avoids circular wrap-around.
        ref = ctx.PilotWaveform;
        n = 2^nextpow2(L + numel(ref));
        r = ifft(fft(rxWaveform, n) .* conj(fft(ref, n)));
        metric = sum(abs(r(1:maxLag+1, :)).^2, 2);
        [~, peak] = max(metric);
        offset = peak - 1;
end
assert(offset >= 0 && offset <= maxLag, 'Timing offset %d is outside 0..%d.', offset, maxLag);
slot = rxWaveform(offset + (1:ctx.SlotLength), :);
end
