function f = flow(name)
%FLOW The receiver flows: which networks they use and how they connect.
%   f = nn.flow(name), name one of
%     'fcnn'       rx.sync -> FFT -> features -> FCNN              (DeepRx)
%     'tcnn_feq'   rx.sync -> TCNN -> FFT -> LS -> interpolate -> LMMSE eq. -> demap
%     'tcnn_fcnn'  rx.sync -> TCNN -> FFT -> features -> FCNN       (hybrid)
%   f.Networks: names of the networks (fields of params / model.Nets).
%   f.Forward:  @(params, batch, training, ctx) -> [logits, states], the
%               forward pass nn.train and nn.receive call.
%   f.SinglePilotLayout: true if the flow takes the pilot layout from ctx
%               (conventional blocks), so all slots must share one layout.
%   A new flow is one more case here plus its forward function.
switch name
    case 'fcnn'
        f.Networks = {'fcnn'};
        f.Forward = @nn.fcnn_forward;
        f.SinglePilotLayout = false;
    case 'tcnn_feq'
        f.Networks = {'tcnn'};
        f.Forward = @nn.flow_tcnn_feq;
        f.SinglePilotLayout = true;
    case 'tcnn_fcnn'
        f.Networks = {'tcnn', 'fcnn'};
        f.Forward = @nn.flow_tcnn_fcnn;
        f.SinglePilotLayout = false;
    otherwise
        error('Unknown receiver flow: %s.', name);
end
f.Name = name;
end
