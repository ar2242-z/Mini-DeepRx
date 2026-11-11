function [llr, trace] = receive(rxWaveform, ctx, model)
%RECEIVE Neural receiver: received samples to bit LLRs.
%   [llr, trace] = nn.receive(rxWaveform, ctx, model)
%   The counterpart of rx.receive with the same inputs and outputs. The
%   chain follows model.Flow (see nn.flow):
%     'fcnn'       rx.sync -> rx.ofdm_demod -> nn.deeprx_features -> FCNN
%     'tcnn_feq'   rx.sync -> TCNN -> conventional receiver (FFT ... demap)
%     'tcnn_fcnn'  rx.sync -> TCNN -> rx.ofdm_demod -> features -> FCNN
%     'ftf'        rx.sync -> rx.ofdm_demod -> features -> FTF-CNN
%   It runs the same forward pass as in training, in prediction mode.
%   Uses only rxWaveform and ctx, never the truth. ctx must have the pilot
%   layout and link the model was trained for.
%   model: saved by train_receiver (Flow, Nets). llr: (Q*D) x 1,
%   log P(0)/P(1), ready for sim.evaluate / rx.ldpc_decode.
assert(isfield(model, 'Flow') && isfield(model, 'Nets'), ...
    'The model has no Flow and Nets; it is not from train_receiver.');
flow = nn.flow(model.Flow);
if isfield(model.Nets, 'tcnn')
    expected = nn.tcnn_input_size(ctx);
    assert(isequal(model.TCNNInputSize, expected), ...
        'This link gives TCNN input %s; the model expects %s.', ...
        mat2str(expected), mat2str(model.TCNNInputSize));
end
[slot, trace.Offset] = rx.sync(rxWaveform, ctx);
batch.Slots = dlarray(single(slot));
batch.PilotGrid = ctx.PilotGrid;
trace.Logits = double(extractdata(flow.Forward(model.Nets, batch, false, ctx)));
llr = nn.to_llr(trace.Logits, ctx);
end
