function net = tcnn_paper(inputSize, widths)
%TCNN_PAPER The time-domain CNN of TCNN/FEQ and HybridDeepRx.
%   net = nn.tcnn_paper(inputSize) builds the network of Pihlajasalo et
%   al., "Deep Learning OFDM Receivers for Improved Power Efficiency and
%   Coverage", IEEE TWC 22(8), 2023, Sec. III-C, for input [Lmax N 2]: one
%   antenna's slot cut into its N OFDM symbols, each symbol a column of
%   Lmax = longest CP + Nfft samples (CP included, shorter CPs zero-padded
%   in front), real and imaginary parts as the two channels.
%   nn.tcnn_apply makes this layout and runs every antenna through the same
%   network. net = nn.tcnn_paper(inputSize, widths) changes the blocks.
%   From the paper: five pre-activation ResNet blocks with 64-128-256-128-64
%   filters of size 3x3 over (sample, symbol), output the same size as the
%   input, last layer a 1x1 convolution to 2 channels with no activation.
%   The network output IS the cleaned signal (not input + correction), so
%   unlike nn.tcnn an untrained network does not pass the samples through.
%   Not given in the paper, chosen here as in DeepRx (Honkala et al.,
%   2021), whose blocks the paper says it follows: each block is twice
%   BN -> ReLU -> depthwise 3x3 -> pointwise 1x1 (depthwise-separable,
%   depth multiplier 1), a 1x1 projection on the skip path when the width
%   changes, a 3x3 input convolution to the first width, and BN -> ReLU
%   before the last layer. With the default widths this gives about 293 k
%   parameters, close to the paper's "about 280 thousand" for TCNN/FEQ
%   (plain 3x3 convolutions would give about 2.4 M). Returns an
%   initialized dlnetwork (SSCB).
if nargin < 2, widths = [64 128 256 128 64]; end

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
        groupedConvolution2dLayer(3, 1, 'channel-wise', 'Padding', 'same', 'Name', [p 'depth1'])
        convolution2dLayer(1, widths(b), 'Name', [p 'point1'])
        batchNormalizationLayer('Name', [p 'bn2'])
        reluLayer('Name', [p 'relu2'])
        groupedConvolution2dLayer(3, 1, 'channel-wise', 'Padding', 'same', 'Name', [p 'depth2'])
        convolution2dLayer(1, widths(b), 'Name', [p 'point2'])
        additionLayer(2, 'Name', [p 'add'])]);
    net = connectLayers(net, previous, [p 'bn1']);
    if widths(b) == previousWidth
        net = connectLayers(net, previous, [p 'add/in2']);
    else
        net = addLayers(net, convolution2dLayer(1, widths(b), 'Name', [p 'skip']));
        net = connectLayers(net, previous, [p 'skip']);
        net = connectLayers(net, [p 'skip'], [p 'add/in2']);
    end
    previous = [p 'add']; previousWidth = widths(b);
end
net = addLayers(net, [
    batchNormalizationLayer('Name', 'output_bn')
    reluLayer('Name', 'output_relu')
    convolution2dLayer(1, inputSize(3), 'Name', 'output')]);
net = connectLayers(net, previous, 'output_bn');
net = initialize(net);
end
