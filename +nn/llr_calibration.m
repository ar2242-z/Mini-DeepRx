function net = llr_calibration()
%LLR_CALIBRATION Two learnable numbers that calibrate conventional LLRs.
%   net = nn.llr_calibration() returns a tiny dlnetwork holding
%   z = [log a; log eps], used by nn.calibrate_llr:
%       LLR' = a * LLR * nu / (nu + eps)
%   i.e. the demapper's noise variance nu becomes (nu + eps)/a:
%     eps  an error floor relative to the unit symbol power: what nu leaves
%          out, chiefly the channel estimation error of one DMRS symbol held
%          over a slot with Doppler, and PA distortion. It caps |LLR| at
%          high SNR, where the conventional LLRs are most overconfident.
%     a    an overall scale (e.g. for the max-log approximation).
%   Both are positive, so the hard decisions are those of the conventional
%   receiver: calibration changes the confidence, never the bits.
%   Why: trained through an overconfident receiver, a TCNN lowers the loss
%   fastest by making the signal look noisier (smaller LLRs), which raises
%   the BER. With the calibration learned alongside, confidence is fixed
%   here and the TCNN's gradient must come from getting more bits right.
%   Stored as a 1x1 convolution (Weights 1x1x1x2, Bias 1x1x2) so nn.train
%   updates, moves and saves it like any network; z = Gain*W + b with
%   Gain = 50, so Adam's per-step change of W moves z 50 times as fast and
%   the calibration settles within a few hundred steps.
%   Starts at a = 1, eps = 1e-3.
gain = nn.calibrate_llr();
z0 = [0; log(1e-3)];
net = dlnetwork([
    imageInputLayer([1 1 1], 'Normalization', 'none', 'Name', 'input')
    convolution2dLayer(1, 2, 'Weights', reshape(z0/gain, 1, 1, 1, 2), ...
        'Bias', zeros(1, 1, 2), 'Name', 'calibration')]);
end
