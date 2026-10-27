function [planes, mask] = labels(codedBits, ctx)
%LABELS Training targets: coded bits as bit planes on the resource grid.
%   [planes, mask] = nn.labels(codedBits, ctx)
%   codedBits: (Q*D) x B bits on the grid (truth.CodedBits), transmit order,
%   D = numel(ctx.DataIndices). planes: F x N x Q x B single, plane b holds
%   bit b of the symbol on each data RE (16QAM: 1 real sign, 2 imaginary
%   sign, 3 real level, 4 imaginary level), zero elsewhere. mask: logical,
%   true on data REs only; pilots and empty REs carry no loss.
%   nn.to_llr inverts the layout.
Q = ctx.BitsPerSymbol;
F = ctx.NumSubcarriers; N = ctx.NumSymbols;
D = numel(ctx.DataIndices);
B = size(codedBits, 2);
assert(size(codedBits, 1) == Q*D, 'Expected %d coded bits per slot, got %d.', Q*D, size(codedBits, 1));
bitsByPlane = permute(reshape(single(codedBits), Q, D, B), [2 1 3]); % D x Q x B
planes = zeros(F*N, Q, B, 'single');
planes(ctx.DataIndices, :, :) = bitsByPlane;
planes = reshape(planes, F, N, Q, B);
mask = repmat(ctx.DataMask, 1, 1, Q, B);
end
