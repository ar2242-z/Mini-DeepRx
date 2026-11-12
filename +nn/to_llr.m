function llr = to_llr(scores, ctx)
%TO_LLR Network output planes to LLRs in transmit bit order.
%   llr = nn.to_llr(scores, ctx)
%   scores: F x N x (Q*L) x B network outputs (numeric or dlarray), the
%   logit of P(bit = 1) for each bit plane, laid out as nn.labels (the Q
%   planes of layer 1, then those of layer 2, ...).
%   llr: (Q*L*D) x B, LLR = log P(b=0)/P(b=1) = -logit, the convention of
%   rx.demap_llr and rx.ldpc_decode. This is the only place where the
%   network's convention is converted.
if isa(scores, 'dlarray')
    scores = stripdims(scores);
end
[F, N, Q, B] = size(scores);
scores = reshape(scores, F*N, Q, B);
data = scores(ctx.DataIndices, :, :);            % D x (Q*L) x B
llr = -reshape(permute(data, [2 1 3]), [], B);   % (Q*L) x D x B -> (Q*L*D) x B
end
