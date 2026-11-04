function llr = demap_llr(xhat, nu, modulation)
%DEMAP_LLR Soft demapper: equalized symbols to bit log-likelihood ratios.
%   llr = rx.demap_llr(xhat, nu, modulation), xhat and nu are D x B
%   (from rx.equalize). llr is (Q*D) x B in transmit bit order: the Q bits
%   of symbol 1, then of symbol 2, ... (Q = 2 for QPSK, 4 for 16QAM, 6
%   for 64QAM).
%   Convention: LLR = log P(b=0)/P(b=1); positive means bit 0.
%   Max-log approximation with complex Gaussian noise of variance nu:
%       LLR = (min |x-s|^2 over s with b=1 - min over s with b=0) / nu
%   5G mapping (TS 38.211 5.1) separates into the real and imaginary
%   axes: bit 1 sets the sign of the real part (0 -> positive), bit 2 the
%   sign of the imaginary part; for 16QAM bits 3 and 4 select the inner
%   (0) or outer (1) level on the real and imaginary axes; for 64QAM
%   bits 3-6 are two such level bits per axis, alternating real and
%   imaginary.
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
    case '64QAM'
        [r1, r2, r3] = pam8(xr); [i1, i2, i3] = pam8(xi);
        planes = cat(3, r1./nu, i1./nu, r2./nu, i2./nu, r3./nu, i3./nu);
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

function [signLLR, b2LLR, b3LLR] = pam8(x)
    % One axis of 64QAM (TS 38.211 5.1.4): level (1-2b1)*(4-(1-2b2)*(2-(1-2b3)))
    % /sqrt(42), so the bit triples (b1 b2 b3) sit at +3 000, +1 001, +5 010,
    % +7 011, and the same with b1 = 1 on the negative side.
    s = 1/sqrt(42);
    d = @(level) (x - level*s).^2;
    dp1 = d(1); dp3 = d(3); dp5 = d(5); dp7 = d(7);
    dm1 = d(-1); dm3 = d(-3); dm5 = d(-5); dm7 = d(-7);
    pos = min(min(dp1, dp3), min(dp5, dp7)); neg = min(min(dm1, dm3), min(dm5, dm7));
    signLLR = neg - pos;                                         % bit 1 means negative.
    inner = min(min(dp1, dp3), min(dm1, dm3)); outer = min(min(dp5, dp7), min(dm5, dm7));
    b2LLR = outer - inner;                                       % bit 1 means |level| 5 or 7.
    mid = min(min(dp3, dp5), min(dm3, dm5)); edge = min(min(dp1, dp7), min(dm1, dm7));
    b3LLR = edge - mid;                                          % bit 1 means |level| 1 or 7.
end
