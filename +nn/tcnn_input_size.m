function inputSize = tcnn_input_size(ctx, architecture)
%TCNN_INPUT_SIZE Network input size of a TCNN for this link.
%   inputSize = nn.tcnn_input_size(ctx, architecture)
%     'residual'  [SlotLength 1 2*Nr]: the whole slot, all antennas (nn.tcnn)
%     'paper'     [max CP + Nfft, N, 2]: one antenna, one column per OFDM
%                 symbol including its CP (nn.tcnn_paper)
switch architecture
    case 'residual'
        inputSize = [ctx.SlotLength 1 2*ctx.NumReceiveAntennas];
    case 'paper'
        inputSize = [max(ctx.CyclicPrefixLengths) + ctx.Nfft, ctx.NumSymbols, 2];
    otherwise
        error('Unknown TCNN architecture: %s.', architecture);
end
end
