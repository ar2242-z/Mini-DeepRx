function inputSize = tcnn_input_size(ctx)
%TCNN_INPUT_SIZE Network input size of the TCNN (nn.tcnn) for this link.
%   inputSize = nn.tcnn_input_size(ctx) is [SlotLength 1 2*Nr]: the whole
%   synced slot, real then imaginary parts of all antennas as channels.
inputSize = [ctx.SlotLength 1 2*ctx.NumReceiveAntennas];
end
