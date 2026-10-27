function nets = model_nets(model)
%MODEL_NETS The networks of a saved model, one field per network.
%   nets = nn.model_nets(model) returns model.Nets (train_receiver), or
%   struct('fcnn', model.Net) for FCNN models from train_fcnn and the
%   legacy model, which have no Flow.
if isfield(model, 'Nets')
    nets = model.Nets;
else
    nets = struct('fcnn', model.Net);
end
end
