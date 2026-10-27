function net = tcnn(inputSize, widths, dilations)
%TCNN Time-domain CNN that cleans the received samples before the FFT.
%   net = nn.tcnn(inputSize) builds the network for input [L 1 2*Nr]: the
%   L samples of a synced slot along the first axis, real parts of the Nr
%   antennas then imaginary parts as channels (nn.tcnn_apply makes it).
%   net = nn.tcnn(inputSize, widths, dilations) changes the residual
%   blocks: widths(b) channels and time dilation dilations(b) for block b.
%   Default widths 64-128-256-128-64, as the TCNN of Pihlajasalo et al.,
%   IEEE TWC 2023, Sec. III-C. The whole-slot 1-D layout, dilations, depth
%   multiplier 2 and the input + correction output are this project's
%   choices; nn.tcnn_paper is the paper's per-symbol 2-D network.
%   Structure: 3-tap input convolution along time, then residual blocks of
%     BN -> ReLU -> depthwise 3-tap (dilated, depth multiplier 2) -> 1x1
%     BN -> ReLU -> depthwise 3-tap (dilated)                      -> 1x1
%   then a 1x1 layer back to 2*Nr channels that is ADDED to the input: the
%   network outputs input + correction. Physically, the PA distorts each
%   transmitted sample on its own, and the channel then spreads it over a
%   few tens of samples (the delay spread); default dilations 1-2-4-2-1
%   give each output a view of 42 samples (5.5 us) around it.
%   The correction layer starts at ZERO, so the untrained TCNN passes the
%   samples through unchanged and the receiver behind it starts exactly
%   as it would without a TCNN. Returns an initialized dlnetwork (SSCB).
if nargin < 2
    widths = [64 128 256 128 64];
    dilations = [1 2 4 2 1];
end
assert(numel(dilations) == numel(widths), 'Give one dilation per width.');

net = dlnetwork;
net = addLayers(net, [
    imageInputLayer(inputSize, 'Normalization', 'none', 'Name', 'input')
    convolution2dLayer([3 1], widths(1), 'Padding', 'same', 'Name', 'input_conv')]);
previous = 'input_conv'; previousWidth = widths(1);
for b = 1:numel(widths)
    p = sprintf('block%d_', b);
    d = [dilations(b) 1];
    net = addLayers(net, [
        batchNormalizationLayer('Name', [p 'bn1'])
        reluLayer('Name', [p 'relu1'])
        groupedConvolution2dLayer([3 1], 2, 'channel-wise', 'Padding', 'same', ...
            'DilationFactor', d, 'Name', [p 'depth1'])
        convolution2dLayer(1, widths(b), 'Name', [p 'point1'])
        batchNormalizationLayer('Name', [p 'bn2'])
        reluLayer('Name', [p 'relu2'])
        groupedConvolution2dLayer([3 1], 2, 'channel-wise', 'Padding', 'same', ...
            'DilationFactor', d, 'Name', [p 'depth2'])
        convolution2dLayer(1, widths(b), 'Name', [p 'point2'])
        additionLayer(2, 'Name', [p 'add'])]);
    net = connectLayers(net, previous, [p 'bn1']);
    if widths(b) == previousWidth
        net = connectLayers(net, previous, [p 'add/in2']);
    else
        % 1x1 projection so the skip path matches the new width.
        net = addLayers(net, convolution2dLayer(1, widths(b), 'Name', [p 'skip']));
        net = connectLayers(net, previous, [p 'skip']);
        net = connectLayers(net, [p 'skip'], [p 'add/in2']);
    end
    previous = [p 'add']; previousWidth = widths(b);
end
net = addLayers(net, [
    convolution2dLayer(1, inputSize(3), 'WeightsInitializer', 'zeros', ...
        'BiasInitializer', 'zeros', 'Name', 'correction')
    additionLayer(2, 'Name', 'output')]);
net = connectLayers(net, previous, 'correction');
net = connectLayers(net, 'input', 'output/in2');
net = initialize(net);
end
