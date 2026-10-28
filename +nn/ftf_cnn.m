function nets = ftf_cnn(inputSize, numBits, M, widths, dilations, split, timeDilations)
%FTF_CNN Frequency-domain HybridDeepRx (FTF-CNN) networks.
%   nets = nn.ftf_cnn(inputSize, numBits, M) builds the three networks of
%   Pihlajasalo et al., TWC 2023, Sec. III-E and Fig. 2(b), for DeepRx
%   features of size [F N C] (nn.deeprx_features), numBits output planes
%   and an M-point pseudo-time domain:
%     nets.ftf_f1  frequency: 3x3 input conv, then j residual blocks  F x N
%     -- parallel IFFTs (nn.pair_transform with nn.pseudo_time_basis) --
%     nets.ftf_t   pseudo-time: k residual blocks                     M x N
%     -- parallel FFTs back to the occupied subcarriers --
%     nets.ftf_f2  frequency: l residual blocks, 3x3 output conv      F x N
%   f1 can equalize the channel, t undo the PA in (pseudo-)time where it
%   acted, and f2 map the symbols to bit logits (logit of P(bit = 1)).
%   nets = nn.ftf_cnn(inputSize, numBits, M, widths, dilations, split, timeDilations)
%   changes the blocks: widths and [frequency time] dilations as nn.fcnn
%   (default nn.fcnn_layout), split = [j k l] blocks per part (default
%   [4 3 4], sum = numel(widths)), and the pseudo-time blocks' [sample
%   symbol] dilations timeDilations (k x 2, default ones: the PA has no
%   memory). The blocks are DeepRx's (nn.resnet_blocks), so with the same
%   widths the FTF-CNN has exactly the FCNN's parameters. The widths at the
%   two transforms must be even (real/imaginary pairs). Each network is an
%   initialized dlnetwork (format SSCB).
if nargin < 4 || isempty(widths)
    [widths, dilations] = nn.fcnn_layout();
end
if nargin < 6 || isempty(split), split = [4 3 4]; end
if nargin < 7 || isempty(timeDilations), timeDilations = ones(split(2), 2); end
assert(numel(split) == 3 && all(split >= 1) && sum(split) == numel(widths), ...
    'split must give [j k l] >= 1 blocks summing to the %d widths.', numel(widths));
assert(isequal(size(timeDilations), [split(2) 2]), 'timeDilations must be %d x 2.', split(2));
assert(size(dilations,1) == numel(widths) && size(dilations,2) == 2, ...
    'dilations must have one [frequency time] row per width.');
f1 = 1:split(1); t = split(1) + (1:split(2)); f2 = split(1) + split(2) + (1:split(3));
assert(mod(widths(f1(end)), 2) == 0 && mod(widths(t(end)), 2) == 0, ...
    'The widths before each transform (%d, %d) must be even.', widths(f1(end)), widths(t(end)));
[F, N] = deal(inputSize(1), inputSize(2));

net = dlnetwork;
net = addLayers(net, [
    imageInputLayer(inputSize, 'Normalization', 'none', 'Name', 'input')
    convolution2dLayer(3, widths(1), 'Padding', 'same', 'Name', 'input_conv')]);
net = nn.resnet_blocks(net, 'input_conv', widths(1), widths(f1), dilations(f1,:));
nets.ftf_f1 = initialize(net);

net = dlnetwork;
net = addLayers(net, imageInputLayer([M N widths(f1(end))], 'Normalization', 'none', 'Name', 'input'));
net = nn.resnet_blocks(net, 'input', widths(f1(end)), widths(t), timeDilations);
nets.ftf_t = initialize(net);

net = dlnetwork;
net = addLayers(net, imageInputLayer([F N widths(t(end))], 'Normalization', 'none', 'Name', 'input'));
[net, previous] = nn.resnet_blocks(net, 'input', widths(t(end)), widths(f2), dilations(f2,:));
net = addLayers(net, convolution2dLayer(3, numBits, 'Padding', 'same', 'Name', 'bit_logits'));
net = connectLayers(net, previous, 'bit_logits');
nets.ftf_f2 = initialize(net);
end
