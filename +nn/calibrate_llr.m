function llr = calibrate_llr(net, llr, nu)
%CALIBRATE_LLR Apply the learned LLR calibration (see nn.llr_calibration).
%   llr = nn.calibrate_llr(net, llr, nu)
%   llr: (Q*D) x B from rx.demap_llr, nu: D x B from rx.equalize.
%       LLR' = a * LLR * nu / (nu + eps),  [log a; log eps] = Gain*W + b
%   Max-log LLRs are proportional to 1/nu, so this is exactly the demapper
%   run with noise variance (nu + eps)/a. Reads the network's learnables
%   directly (no forward pass), so it runs on CPU or GPU and dlgradient
%   reaches them. gain = nn.calibrate_llr() returns Gain.
gain = 50;
if nargin == 0
    llr = gain;
    return
end
L = net.Learnables;
W = L.Value{strcmp(L.Parameter, 'Weights')};
b = L.Value{strcmp(L.Parameter, 'Bias')};
z = gain*W(:) + b(:);
a = exp(z(1)); floorVariance = exp(z(2));
factor = a .* nu ./ (nu + floorVariance);                     % D x B
[QD, B] = size(llr); D = size(nu, 1);
llr = reshape(reshape(llr, QD/D, D, B) .* reshape(factor, 1, D, B), QD, B);
end
