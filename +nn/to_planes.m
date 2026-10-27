function scores = to_planes(scores, ctx)
%TO_PLANES Per-bit scores in transmit order onto the grid as bit planes.
%   planes = nn.to_planes(scores, ctx), the inverse of nn.to_llr's layout:
%   scores (Q*D) x B in transmit bit order (e.g. -LLR from rx.demap_llr,
%   the logit of P(bit = 1)) -> F x N x Q x B, laid out as nn.labels, zero
%   off the data REs. Lets a conventional demapper's output be trained with
%   the same loss and labels as a network's. Runs on dlarray.
Q = ctx.BitsPerSymbol;
D = numel(ctx.DataIndices);
B = size(scores, 2);
byPlane = permute(reshape(scores, Q, D, B), [2 1 3]);        % D x Q x B
planes = zeros(ctx.NumSubcarriers*ctx.NumSymbols, Q, B, 'like', byPlane);
planes(ctx.DataIndices, :, :) = byPlane;
scores = reshape(planes, ctx.NumSubcarriers, ctx.NumSymbols, Q, B);
end
