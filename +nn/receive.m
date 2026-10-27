function [llr, trace] = receive(rxWaveform, ctx, model)
%RECEIVE Neural receiver: received samples to bit LLRs.
%   [llr, trace] = nn.receive(rxWaveform, ctx, model)
%   The counterpart of rx.receive with the same inputs and outputs. The
%   chain follows model.Flow (see nn.flow); models without a Flow (from
%   train_fcnn, and the legacy model) are FCNNs:
%     'fcnn'       rx.sync -> rx.ofdm_demod -> nn.deeprx_features -> FCNN
%     'tcnn_feq'   rx.sync -> TCNN -> conventional receiver (FFT ... demap)
%     'tcnn_fcnn'  rx.sync -> TCNN -> rx.ofdm_demod -> features -> FCNN
%   Uses only rxWaveform and ctx, never the truth. ctx must have the pilot
%   layout and link the model was trained for.
%   model: saved by train_receiver / train_fcnn. llr: (Q*D) x 1,
%   log P(0)/P(1), ready for evaluate / rx.ldpc_decode.
flowName = 'fcnn';
if isfield(model, 'Flow'), flowName = model.Flow; end
[slot, trace.Offset] = rx.sync(rxWaveform, ctx);
if strcmp(flowName, 'fcnn')
    assert(strcmp(model.FeatureLayout, 'deeprx-v1'), ...
        'Model expects feature layout %s; this receiver builds deeprx-v1.', model.FeatureLayout);
    trace.Y = rx.ofdm_demod(slot, ctx);
    Z = dlarray(nn.deeprx_features(trace.Y, ctx.PilotGrid), 'SSCB');
    assert(isequal(size(Z, 1:3), model.InputSize), 'Input %s does not match the network input %s.', ...
        mat2str(size(Z, 1:3)), mat2str(model.InputSize));
    trace.Logits = double(extractdata(predict(model.Net, Z)));
else
    % The same forward pass as in training, in prediction mode.
    flow = nn.flow(flowName);
    nets = nn.model_nets(model);
    if isfield(nets, 'tcnn')
        architecture = 'residual';  % Models saved before the paper TCNN.
        if isfield(model, 'TCNNArchitecture'), architecture = model.TCNNArchitecture; end
        expected = nn.tcnn_input_size(ctx, architecture);
        assert(isequal(model.TCNNInputSize, expected), ...
            'This link gives TCNN input %s; the model expects %s.', ...
            mat2str(expected), mat2str(model.TCNNInputSize));
    end
    batch.Slots = dlarray(single(slot));
    batch.PilotGrid = ctx.PilotGrid;
    trace.Logits = double(extractdata(flow.Forward(nets, batch, false, ctx)));
end
llr = nn.to_llr(trace.Logits, ctx);
end
