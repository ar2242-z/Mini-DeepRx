function [planes, mask] = labels(codedBits, ctx)
%LABELS Training targets: coded bits as bit planes on the resource grid.
%   [planes, mask] = nn.labels(codedBits, ctx)
%   codedBits: (Q*L*D) x B bits on the grid (truth.CodedBits), transmit
%   order, D = numel(ctx.DataIndices), L = ctx.NumLayers.
%   planes: F x N x (Q*L) x B single. Plane Q*(l-1)+b holds bit b of the
%   symbol that layer l carries on each data RE (16QAM: 1 real sign, 2
%   imaginary sign, 3 real level, 4 imaginary level), zero elsewhere: the Q
%   planes of layer 1, then those of layer 2. One layer: F x N x Q x B.
%   mask: logical, true on data REs only; pilots and empty REs carry no loss.
%   nn.to_llr inverts the layout.
Q = ctx.BitsPerSymbol; L = ctx.NumLayers;
F = ctx.NumSubcarriers; N = ctx.NumSymbols;
D = numel(ctx.DataIndices);
B = size(codedBits, 2);
assert(size(codedBits, 1) == Q*L*D, 'Expected %d coded bits per slot, got %d.', Q*L*D, size(codedBits, 1));
% Transmit order is bit, then layer, then data RE (tx.transmit).
bitsByPlane = permute(reshape(single(codedBits), Q*L, D, B), [2 1 3]); % D x (Q*L) x B
planes = zeros(F*N, Q*L, B, 'single');
planes(ctx.DataIndices, :, :) = bitsByPlane;
planes = reshape(planes, F, N, Q*L, B);
mask = repmat(ctx.DataMask, 1, 1, Q*L, B);
end
