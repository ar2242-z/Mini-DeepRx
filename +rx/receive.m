function [llr, trace] = receive(rxWaveform, ctx, mode, genie)
%RECEIVE The conventional receiver: received samples to bit LLRs.
%   [llr, trace] = rx.receive(rxWaveform, ctx, mode)
%   [llr, trace] = rx.receive(rxWaveform, ctx, mode, genie)
%   mode:
%     'conventional'  sync -> CP removal + FFT -> LS at pilots -> linear
%                     interpolation -> LMMSE equalization -> soft demap
%                     (the DMRS-based LMMSE baseline of Pihlajasalo et
%                     al., IEEE TWC 2023, Sec. V)
%     'knownH'        the same, but with the TRUE channel genie.H in place of
%                     estimation and interpolation: the upper bound for any
%                     channel estimator. genie.H is defined at the true
%                     timing (a one-sample shift rotates subcarrier k by
%                     2*pi*k/Nfft), so this mode always uses the true timing
%                     genie.TimingOffset. The noise power is still estimated
%                     from the pilots. With a PA, the data sees g*H, where
%                     g = genie.PAGain is the PA's Bussgang gain (y = g*x +
%                     distortion); a perfect estimator would measure g*H at
%                     the pilots, so this mode uses g*H too. The distortion
%                     itself stays an impairment.
%   genie (truth from run_link) is needed only for 'knownH' and for
%   ctx.TimingEstimation 'perfect'. The conventional receiver with 'dmrs'
%   timing uses nothing but rxWaveform and ctx.
%   llr: (Q*D) x 1 LLRs, log P(0)/P(1), in transmit bit order; pass to
%   rx.ldpc_decode when coded.
%   trace: Offset, Y, Hp, H, Sigma2, Xhat, Nu (the receiver checkpoints).
if nargin < 4
    genie = struct();
end
if strcmp(mode, 'knownH')
    ctx.TimingEstimation = 'perfect';
end
if strcmp(ctx.TimingEstimation, 'perfect')
    assert(isfield(genie, 'TimingOffset'), 'Perfect timing needs genie.TimingOffset.');
    [slot, trace.Offset] = rx.sync(rxWaveform, ctx, genie.TimingOffset);
else
    [slot, trace.Offset] = rx.sync(rxWaveform, ctx);
end
Y = rx.ofdm_demod(slot, ctx);
[Hp, sigma2] = rx.ls_estimate(Y, ctx);
switch mode
    case 'conventional'
        H = rx.interpolate(Hp, ctx);
    case 'knownH'
        assert(isfield(genie, 'H'), 'knownH needs the true channel genie.H.');
        H = genie.H;
        if isfield(genie, 'PAGain'), H = H * genie.PAGain; end
    otherwise
        error('Unknown receiver mode: %s.', mode);
end
[xhat, nu] = rx.equalize(Y, H, sigma2, ctx);
llr = rx.demap_llr(xhat, nu, ctx.Modulation);

trace.Y = Y;
trace.Hp = Hp;
trace.H = H;
trace.Sigma2 = sigma2;
trace.Xhat = xhat;
trace.Nu = nu;
end
