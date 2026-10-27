function Z = deeprx_features(Y, pilotGrid)
%DEEPRX_FEATURES DeepRx network input from the received grid and known pilots.
%   Z = nn.deeprx_features(Y, pilotGrid)
%   Y: F x N x Nr x B received grid (rx.ofdm_demod). pilotGrid: F x N
%   (ctx.PilotGrid) or F x N x 1 x B for a batch that mixes pilot layouts.
%   Z: F x N x 2*(2*Nr+1) x B single, real parts then imaginary parts of
%       [Y, Xp, Hraw],  Hraw = Y .* conj(Xp)
%   Xp is the known pilot grid (zero on data REs) and Hraw the raw LS
%   channel at the pilots (zero elsewhere, NOT interpolated): the network
%   learns interpolation, equalization and demapping itself. For Nr = 2
%   the ten channels are real(Y1,Y2,Xp,H1,H2), imag(Y1,Y2,Xp,H1,H2), the
%   same order as the earlier deeprx_sample.m. Uses only the received
%   signal and the configuration. Products only: runs on dlarray, so a
%   network in front of it (TCNN hybrid) can be trained through it.
B = size(Y, 4);
if size(pilotGrid, 4) == 1 && B > 1
    pilotGrid = repmat(pilotGrid, 1, 1, 1, B);
end
Hraw = Y .* conj(pilotGrid);
Zc = cat(3, Y, pilotGrid, Hraw);
Z = single(cat(3, real(Zc), imag(Zc)));
end
