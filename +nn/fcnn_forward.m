function [logits, states] = fcnn_forward(params, batch, training, ctx)
%FCNN_FORWARD FCNN receiver from synced slot samples to bit logits.
%   [logits, states] = nn.fcnn_forward(params, batch, training, ctx)
%     rx.ofdm_demod -> nn.deeprx_features -> params.fcnn
%   batch.Slots: SlotLength x Nr x B; batch.PilotGrid: F x N x 1 x B.
%   training true (or a cell of names containing 'fcnn', see
%   nn.in_training) uses forward (updates batch-norm statistics, returned
%   in states.fcnn); otherwise predict. logits: F x N x Q x B, unformatted.
%   ctx supplies the FFT window only, which is the same for every pilot
%   layout. This is the forward function nn.train expects.
Y = rx.ofdm_demod(batch.Slots, ctx);
Z = dlarray(nn.deeprx_features(Y, batch.PilotGrid), 'SSCB');
states = struct();
if nn.in_training(training, 'fcnn')
    [logits, states.fcnn] = forward(params.fcnn, Z);
else
    logits = predict(params.fcnn, Z);
end
logits = stripdims(logits);
end
