function [grads, norms] = clip_gradients(grads, maxNorm)
%CLIP_GRADIENTS Scale each network's gradients down to a maximum L2 norm.
%   [grads, norms] = nn.clip_gradients(grads, maxNorm)
%   grads: struct with one Learnables-style table per network (as from
%   dlgradient). norms: struct with each network's norm BEFORE clipping.
%   A network whose norm exceeds maxNorm has all its gradients multiplied
%   by maxNorm/norm, so its step keeps its direction and only its size is
%   capped. Per network, so one network's large gradients (e.g. the LLR
%   calibration's, scaled by its Gain) never shrink another's step.
%   maxNorm Inf only measures the norms.
norms = struct();
for name = fieldnames(grads).'
    n = name{1};
    total = 0;
    for k = 1:height(grads.(n))
        total = total + sum(extractdata(grads.(n).Value{k}).^2, 'all');
    end
    norms.(n) = sqrt(double(gather(total)));
    if norms.(n) > maxNorm
        scale = maxNorm/norms.(n);
        grads.(n).Value = cellfun(@(g) g*scale, grads.(n).Value, 'UniformOutput', false);
    end
end
end
