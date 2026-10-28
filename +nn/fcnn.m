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
    [widths, dilations] = nn.fcnn_layout();
end
assert(size(dilations,1) == numel(widths) && size(dilations,2) == 2, ...
    'dilations must have one [frequency time] row per width.');

net = dlnetwork;
net = addLayers(net, [
    imageInputLayer(inputSize, 'Normalization', 'none', 'Name', 'input')
    convolution2dLayer(3, widths(1), 'Padding', 'same', 'Name', 'input_conv')]);
[net, previous] = nn.resnet_blocks(net, 'input_conv', widths(1), widths, dilations);
net = addLayers(net, convolution2dLayer(3, numBits, 'Padding', 'same', 'Name', 'bit_logits'));
net = connectLayers(net, previous, 'bit_logits');
net = initialize(net);
end
