function [logits, states] = flow_tcnn_fcnn(params, batch, training, ctx)
%FLOW_TCNN_FCNN Hybrid: TCNN in front of the FCNN.
%   [logits, states] = nn.flow_tcnn_fcnn(params, batch, training, ctx)
%     params.tcnn -> rx.ofdm_demod -> nn.deeprx_features -> params.fcnn
%   The TCNN cleans the time samples (e.g. PA distortion), the FCNN
%   detects the bits on the grid. Either network can be frozen (see
%   nn.in_training), e.g. train a TCNN in front of a trained FCNN.
%   logits: F x N x Q x B. states: batch-norm statistics of the networks
%   run in training mode.
[slot, state] = nn.tcnn_apply(params.tcnn, batch.Slots, nn.in_training(training, 'tcnn'), ctx);
batch.Slots = slot;
[logits, states] = nn.fcnn_forward(params, batch, training, ctx);
if ~isempty(state), states.tcnn = state; end
end
