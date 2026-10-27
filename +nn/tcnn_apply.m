function [slot, state] = tcnn_apply(net, slot, training, ctx)
%TCNN_APPLY Run the TCNN on synced slots: received samples in, cleaned out.
%   [slot, state] = nn.tcnn_apply(net, slot, training, ctx)
%   slot: SlotLength x Nr x B complex (numeric or dlarray), from rx.sync.
%   Each slot is scaled to unit RMS for the network (received samples are
%   tiny: the FFT is unscaled) and scaled back afterwards, so everything
%   after the TCNN sees samples on the usual scale. The scale is a fixed
%   per-slot constant, not learned.
%   The layout follows the network input (see nn.tcnn_input_size):
%     [L 1 2*Nr]      nn.tcnn: the whole slot, all antennas as channels
%     [Lmax N 2]      nn.tcnn_paper: each antenna on its own (antennas go
%                     into the batch), one column per OFDM symbol with its
%                     CP, shorter CPs zero-padded in front (ctx gives the
%                     CP lengths)
%   training true uses forward (state: new batch-norm statistics); false
%   uses predict (state empty).
[L, Nr, B] = size(slot);
power = mean(real(slot).^2 + imag(slot).^2, [1 2]);         % 1 x 1 x B
scale = sqrt(max(gather(extractNumeric(power)), realmin('single')));
x = slot ./ scale;
perSymbol = net.Layers(1).InputSize(2) > 1;
if perSymbol
    [index, Lmax] = symbolIndex(ctx);
    x = cat(1, zeros(1, Nr, B, 'like', x), x);              % Row 1: padding.
    x = reshape(x(index + 1, :, :), Lmax, ctx.NumSymbols, 1, Nr*B);
    Z = dlarray(cat(3, real(x), imag(x)), 'SSCB');
else
    Z = dlarray(cat(3, reshape(real(x), L, 1, Nr, B), reshape(imag(x), L, 1, Nr, B)), 'SSCB');
end
state = [];
if training
    [Z, state] = forward(net, Z);
else
    Z = predict(net, Z);
end
Z = stripdims(Z);
% real() keeps the gradient flowing back into the network real-valued, as
% its convolutions require.
if perSymbol
    c = real(Z(:, :, 1, :)) + 1i*real(Z(:, :, 2, :));      % Lmax x N x 1 x Nr*B
    c = reshape(c, [], Nr, B);
    slot = c(index > 0, :, :) .* scale;                     % Drop the padding.
else
    slot = reshape(real(Z(:, :, 1:Nr, :)) + 1i*real(Z(:, :, Nr+1:end, :)), L, Nr, B) .* scale;
end
end

function [index, Lmax] = symbolIndex(ctx)
    % Lmax x N slot sample numbers (1-based) of each symbol with its CP,
    % 0 for the padding. Column-major order of the nonzero entries is
    % 1..SlotLength, so index > 0 picks the samples back in slot order.
    cp = ctx.CyclicPrefixLengths; Nfft = ctx.Nfft;
    Lmax = max(cp) + Nfft;
    starts = cumsum([0 cp(1:end-1) + Nfft]);
    index = zeros(Lmax, ctx.NumSymbols);
    for n = 1:ctx.NumSymbols
        len = cp(n) + Nfft;
        index(Lmax-len+1:end, n) = starts(n) + (1:len);
    end
end

function x = extractNumeric(x)
    if isa(x, 'dlarray'), x = extractdata(x); end
end
