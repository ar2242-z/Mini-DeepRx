function [slot, state] = tcnn_apply(net, slot, training)
%TCNN_APPLY Run the TCNN on synced slots: received samples in, cleaned out.
%   [slot, state] = nn.tcnn_apply(net, slot, training)
%   slot: SlotLength x Nr x B complex (numeric or dlarray), from rx.sync.
%   Each slot is scaled to unit RMS for the network (received samples are
%   tiny: the FFT is unscaled) and scaled back afterwards, so everything
%   after the TCNN sees samples on the usual scale. The scale is a fixed
%   per-slot constant, not learned.
%   Network input [L 1 2*Nr] (nn.tcnn_input_size): the whole slot, real
%   parts of the antennas then imaginary parts as channels.
%   training true uses forward (state: new batch-norm statistics); false
%   uses predict (state empty).
[L, Nr, B] = size(slot);
power = mean(real(slot).^2 + imag(slot).^2, [1 2]);         % 1 x 1 x B
scale = sqrt(max(gather(extractNumeric(power)), realmin('single')));
x = slot ./ scale;
Z = dlarray(cat(3, reshape(real(x), L, 1, Nr, B), reshape(imag(x), L, 1, Nr, B)), 'SSCB');
state = [];
if training
    [Z, state] = forward(net, Z);
else
    Z = predict(net, Z);
end
Z = stripdims(Z);
% real() keeps the gradient flowing back into the network real-valued, as
% its convolutions require.
slot = reshape(real(Z(:, :, 1:Nr, :)) + 1i*real(Z(:, :, Nr+1:end, :)), L, Nr, B) .* scale;
end

function x = extractNumeric(x)
    if isa(x, 'dlarray'), x = extractdata(x); end
end
