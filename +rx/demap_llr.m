function llr = demap_llr(xhat, nu, modulation)
%DEMAP_LLR Soft demapper: equalized symbols to bit log-likelihood ratios.
%   llr = rx.demap_llr(xhat, nu, modulation), xhat and nu are D x B
%   (from rx.equalize). llr is (Q*D) x B in transmit bit order: the Q bits
%   of symbol 1, then of symbol 2, ... (Q = 2 for QPSK, 4 for 16QAM).
%   Convention: LLR = log P(b=0)/P(b=1); positive means bit 0.
%   Max-log approximation with complex Gaussian noise of variance nu:
%       LLR = (min |x-s|^2 over s with b=1 - min over s with b=0) / nu
%   5G mapping (TS 38.211 5.1) separates into the real and imaginary
%   axes: bit 1 sets the sign of the real part (0 -> positive), bit 2 the
%   sign of the imaginary part; for 16QAM bits 3 and 4 select the inner
%   (0) or outer (1) level on the real and imaginary axes.
%   min and products only: runs on dlarray.
[D, B] = size(xhat);
xr = real(xhat); xi = imag(xhat);
switch modulation
    case 'QPSK'
        a = 1/sqrt(2);
        % (|x+a|^2 - |x-a|^2) per axis = 4*a*x.
        planes = cat(3, 4*a*xr./nu, 4*a*xi./nu);
    case '16QAM'
        [sr, mr] = pam4(xr); [si, mi] = pam4(xi);
        planes = cat(3, sr./nu, si./nu, mr./nu, mi./nu);
end
% D x B x Q -> Q x D x B -> (Q*D) x B, so each symbol's bits are adjacent.
llr = reshape(permute(planes, [3 1 2]), [], B);
end

function [signLLR, levelLLR] = pam4(x)
    % One axis of 16QAM: levels +1, +3, -1, -3 (/sqrt(10)) carry the bit
    % pairs (sign, level) = 00, 01, 10, 11.
    s = 1/sqrt(10);
    d1 = (x - s).^2; d3 = (x - 3*s).^2; dm1 = (x + s).^2; dm3 = (x + 3*s).^2;
    signLLR = min(dm1, dm3) - min(d1, d3);   % bit 1 means negative.
    levelLLR = min(d3, dm3) - min(d1, dm1);  % bit 1 means outer level.
end
