function Z = deeprx_features(Y, pilotGrid)
%DEEPRX_FEATURES DeepRx network input from the received grid and known pilots.
%   Z = nn.deeprx_features(Y, pilotGrid)
%   Y: F x N x Nr x B received grid (rx.ofdm_demod). pilotGrid: the known
%   pilots of the L layers along dimension 5, F x N x 1 x 1 x L
%   (ctx.PilotGrid) or F x N x 1 x B x L for a batch that mixes pilot
%   layouts. One layer: F x N or F x N x 1 x B.
%   Z: F x N x 2*(Nr + L + Nr*L) x B single (nn.fcnn_input_size), real
%   parts then imaginary parts of
%       [Y, Xp, Hraw],  Hraw = Y .* conj(Xp)
%   Xp is the known pilot grid of each layer (zero on data REs) and Hraw
%   the raw LS channel at the pilots from each layer to each antenna (zero
%   elsewhere, NOT interpolated; layers whose pilots share resource
%   elements are NOT separated): the network learns port separation,
%   interpolation, equalization and demapping itself. Hraw is ordered
%   layer by layer: all antennas of layer 1, then of layer 2.
%   For Nr = 2 and one layer the ten channels are real(Y1,Y2,Xp,H1,H2),
%   imag(Y1,Y2,Xp,H1,H2), the same order as the earlier deeprx_sample.m.
%   Uses only the received signal and the configuration. Products only:
%   runs on dlarray, so a network in front of it (TCNN hybrid) can be
%   trained through it.
[F, N, Nr, B] = size(Y);
L = size(pilotGrid, 5);
if size(pilotGrid, 4) == 1 && B > 1
    pilotGrid = repmat(pilotGrid, 1, 1, 1, B);
end
Hraw = Y .* conj(pilotGrid);                              % F x N x Nr x B x L
Xp = reshape(permute(pilotGrid, [1 2 3 5 4]), F, N, L, B);
Hraw = reshape(permute(Hraw, [1 2 3 5 4]), F, N, Nr*L, B);
Zc = cat(3, Y, Xp, Hraw);
Z = single(cat(3, real(Zc), imag(Zc)));
end
