function loss = bce_loss(logits, targets, mask)
%BCE_LOSS Masked binary cross-entropy from logits, averaged over valid bits.
%   loss = nn.bce_loss(logits, targets, mask)
%   logits: logit of P(bit = 1); targets: 0/1; mask: 1 where the bit counts
%   (data REs). All the same size (F x N x Q x B). Per bit:
%       log(1 + exp(s)) - s*t
%   computed stably: no sigmoid layer and no overflow for large |s|. The
%   max-shift form keeps the gradient exact at s = 0: sigmoid(s) - t.
%   With logit = -LLR, minimizing it trains the network to output LLRs.
shift = max(logits, 0);
perBit = shift - logits.*targets + log(exp(-shift) + exp(logits - shift));
loss = sum(perBit.*mask, 'all') / sum(mask, 'all');
end
