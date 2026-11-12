function inputSize = fcnn_input_size(ctx)
%FCNN_INPUT_SIZE Network input size of the FCNN and FTF-CNN for this link.
%   inputSize = nn.fcnn_input_size(ctx) is [F N 2*(Nr + L + Nr*L)], the size
%   of nn.deeprx_features' output for one slot: real then imaginary parts of
%   the received grid (Nr antennas), the known pilots (L layers) and the raw
%   LS channel (Nr*L). One layer: 2*(2*Nr + 1), ten channels for Nr = 2.
Nr = ctx.NumReceiveAntennas; L = ctx.NumLayers;
inputSize = [ctx.NumSubcarriers ctx.NumSymbols 2*(Nr + L + Nr*L)];
end
