function [logits, states] = flow_ftf(params, batch, training, ctx)
%FLOW_FTF Frequency-domain HybridDeepRx (FTF-CNN), after the FFT only.
%   [logits, states] = nn.flow_ftf(params, batch, training, ctx)
%     rx.ofdm_demod -> nn.deeprx_features -> params.ftf_f1 -> parallel IFFTs
%     -> params.ftf_t (pseudo-time) -> parallel FFTs -> params.ftf_f2
%   (Pihlajasalo et al., TWC 2023, Sec. III-E; networks from nn.ftf_cnn).
%   The front end before the FFT is the ordinary receiver's. The transforms
%   (nn.pair_transform with nn.pseudo_time_basis) are fixed and unitary,
%   so gradients reach f1 through them. Their size M is the t network's
%   input height. training: see nn.in_training. logits: F x N x Q x B,
%   unformatted; states: batch-norm statistics of networks in training.
states = struct();
Y = rx.ofdm_demod(batch.Slots, ctx);
Z = dlarray(nn.deeprx_features(Y, batch.PilotGrid), 'SSCB');
A = single(nn.pseudo_time_basis(ctx, params.ftf_t.Layers(1).InputSize(1)));
if isgpuarray(extractdata(Z)), A = gpuArray(A); end
[Z, states] = run(params, 'ftf_f1', Z, training, states);
Z = dlarray(nn.pair_transform(stripdims(Z), A), 'SSCB');      % IFFTs.
[Z, states] = run(params, 'ftf_t', Z, training, states);
Z = dlarray(nn.pair_transform(stripdims(Z), A'), 'SSCB');     % FFTs back.
[Z, states] = run(params, 'ftf_f2', Z, training, states);
logits = stripdims(Z);
end

function [Z, states] = run(params, name, Z, training, states)
    if nn.in_training(training, name)
        [Z, states.(name)] = forward(params.(name), Z);
    else
        Z = predict(params.(name), Z);
    end
end
