function [net, previous] = resnet_blocks(net, previous, previousWidth, widths, dilations)
%RESNET_BLOCKS Append DeepRx pre-activation residual blocks to a network.
%   [net, previous] = nn.resnet_blocks(net, previous, previousWidth, widths, dilations)
%   adds one block per widths(b), after layer `previous` (previousWidth
%   channels), and returns the name of the last block's output. Block b:
%     BN -> ReLU -> depthwise 3x3 (dilated, depth multiplier 2) -> 1x1
%     BN -> ReLU -> depthwise 3x3 (dilated)                      -> 1x1
%   plus a skip path (a 1x1 projection when the width changes).
%   dilations(b,:) is the [first-dimension second-dimension] dilation.
%   Layers are named block<b>_*. Used by nn.fcnn and nn.ftf_cnn.
assert(size(dilations,1) == numel(widths) && size(dilations,2) == 2, ...
    'dilations must have one two-element row per width.');
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
end
