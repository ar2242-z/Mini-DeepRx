function [widths, dilations] = fcnn_layout()
%FCNN_LAYOUT The DeepRx paper's residual blocks (Honkala et al., Table I).
%   [widths, dilations] = nn.fcnn_layout(): 11 blocks, widths(b) channels
%   and [frequency time] dilation dilations(b,:) for block b (the paper
%   lists (time, frequency)). The default of nn.fcnn and nn.ftf_cnn.
widths = [64 64 128 128 256 256 256 128 128 64 64];
dilations = [1 1; 1 1; 3 2; 3 2; 3 2; 6 3; 3 2; 3 2; 3 2; 1 1; 1 1];
end
