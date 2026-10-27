function [logits, states] = flow_tcnn_feq(params, batch, training, ctx)
%FLOW_TCNN_FEQ TCNN in front of the conventional receiver (TCNN/FEQ).
%   [logits, states] = nn.flow_tcnn_feq(params, batch, training, ctx)
%     params.tcnn -> rx.ofdm_demod -> rx.ls_estimate -> rx.interpolate
%     (linear) -> rx.equalize (LMMSE) -> rx.demap_llr, the paper's FEQ
%     -> nn.calibrate_llr with params.calibration (when present)
%   The TCNN and the two calibration numbers learn; the conventional blocks
%   are fixed, but gradients pass back through them, so the TCNN learns
%   whatever helps the conventional receiver decide the bits. The
%   calibration (nn.llr_calibration) only rescales the LLRs, never flips a
%   bit; it absorbs the receiver's overconfidence, which the TCNN would
%   otherwise "fix" by making the signal look noisier. logits = -LLR (the logit of P(bit = 1)) on the
%   grid, F x N x Q x B, for the same loss and labels as the FCNN.
%   ctx gives the pilot layout: every slot must use it (one PilotConfig).
%   training: see nn.in_training. states.tcnn: new batch-norm statistics.
states = struct();
[slot, state] = nn.tcnn_apply(params.tcnn, batch.Slots, nn.in_training(training, 'tcnn'), ctx);
if ~isempty(state), states.tcnn = state; end
Y = rx.ofdm_demod(slot, ctx);
[Hp, sigma2] = rx.ls_estimate(Y, ctx);
H = rx.interpolate(Hp, ctx);
[xhat, nu] = rx.equalize(Y, H, sigma2, ctx);
llr = rx.demap_llr(xhat, nu, ctx.Modulation);
if isfield(params, 'calibration')   % Models saved before it have none.
    llr = nn.calibrate_llr(params.calibration, llr, nu);
end
logits = nn.to_planes(-llr, ctx);
end
