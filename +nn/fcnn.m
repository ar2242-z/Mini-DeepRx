function net = fcnn(inputSize, numBits, widths, dilations)
%FCNN DeepRx fully convolutional receiver network (Honkala et al., Table I).
%   net = nn.fcnn(inputSize, numBits) builds the paper's network for input
%   [F N C] (from nn.deeprx_features) and numBits output planes (Q).
%   net = nn.fcnn(inputSize, numBits, widths, dilations) changes the
%   residual blocks: widths(b) channels and dilation [frequency time] =
%   dilations(b,:) for block b.
%   Structure: 3x3 input convolution, then residual blocks of
%     BN -> ReLU -> depthwise 3x3 (dilated, depth multiplier 2) -> 1x1
%     BN -> ReLU -> depthwise 3x3 (dilated)                      -> 1x1
%   plus a skip path (a 1x1 projection when the width changes), then a 3x3
%   output convolution. Output: F x N x numBits logits of P(bit = 1).
%   Dilation lets a few layers see far across frequency and time, where
%   the pilots are. Returns an initialized dlnetwork (format SSCB).
if nargin < 3
    widths = [64 64 128 128 256 256 256 128 128 64 64];
    % (frequency, time) dilations; the paper lists (time, frequency).
    dilations = [1 1; 1 1; 3 2; 3 2; 3 2; 6 3; 3 2; 3 2; 3 2; 1 1; 1 1];
end
assert(size(dilations,1) == numel(widths) && size(dilations,2) == 2, ...
    'dilations must have one [frequency time] row per width.');

net = dlnetwork;
net = addLayers(net, [
    imageInputLayer(inputSize, 'Normalization', 'none', 'Name', 'input')
    convolution2dLayer(3, widths(1), 'Padding', 'same', 'Name', 'input_conv')]);
previous = 'input_conv'; previousWidth = widths(1);
for b = 1:numel(widths)
    p = sprintf('block%d_', b);
    net = addLayers(net, [
        batchNormalizationLayer('Name', [p 'bn1'])
        reluLayer('Name', [p 'relu1'])
        groupedConvolution2dLayer(3, 2, 'channel-wise', 'Padding', 'same', ...
            'DilationFactor', dilations(b,:), 'Name', [p 'depth1'])
        convolution2dLayer(1, widths(b), 'Name', [p 'point1'])
        batchNormalizationLayer('Name', [p 'bn2'])
        reluLayer('Name', [p 'relu2'])
        groupedConvolution2dLayer(3, 2, 'channel-wise', 'Padding', 'same', ...
            'DilationFactor', dilations(b,:), 'Name', [p 'depth2'])
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
net = addLayers(net, convolution2dLayer(3, numBits, 'Padding', 'same', 'Name', 'bit_logits'));
net = connectLayers(net, previous, 'bit_logits');
net = initialize(net);
end
