function [llr, trace] = receive(rxWaveform, ctx, mode, genie, options)
%RECEIVE The conventional receiver: received samples to bit LLRs.
%   [llr, trace] = rx.receive(rxWaveform, ctx, mode)
%   [llr, trace] = rx.receive(rxWaveform, ctx, mode, genie)
%   [llr, trace] = rx.receive(rxWaveform, ctx, mode, genie, options)
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
%                     distortion; a scalar for the miniPA, one gain per
%                     subcarrier for the memory polynomial PA); a perfect
%                     estimator would measure g.*H at the pilots, so this
%                     mode uses g.*H too. The distortion itself stays an
%                     impairment.
%     'dpoi'          Digital Post-Inverse (Babaroglu et al., WAMICON 2023),
%                     the conventional receiver-side PA compensation: LS at
%                     the pilots -> delay-domain smoothing (rx.delay_estimate)
%                     -> equalization -> memory polynomial post-inverse in
%                     time, fitted to this slot's pilots (rx.dpoi) -> soft
%                     demap. Uses nothing but rxWaveform and ctx.
%     'smoothed'      'dpoi' without the post-inverse: the same channel
%                     estimate, equalizer and noise estimate. The difference
%                     between 'dpoi' and 'smoothed' is the post-inverse alone.
%     'dpoi_genie'    'dpoi' with the coefficients fitted to the true grid
%                     genie.Grid on every RE instead of the pilots: the best
%                     this post-inverse can do with the estimated channel.
%     'dpoi_knownH'   'dpoi' with the true channel g.*H and true timing, as
%                     'knownH'; the coefficients still come from the pilots.
%   genie (truth from sim.run_link) is needed only for the modes that say so and
%   for ctx.TimingEstimation 'perfect'. The conventional receiver with 'dmrs'
%   timing uses nothing but rxWaveform and ctx.
%   options: struct of rx.dpoi options (Orders, Memory, Oversampling, Ridge)
%   plus DelayWindow and DelayRidge for rx.delay_estimate; DPoI modes only.
%   llr: (Q*L*D) x 1 LLRs, log P(0)/P(1), in transmit bit order; pass to
%   rx.ldpc_decode when coded. Every mode handles one layer (L = 1) for
%   now: rx.ls_estimate and rx.equalize reject more.
%   trace: Offset, Y, Hp, H, Sigma2, Xhat, Nu (the receiver checkpoints),
%   and DPoI (rx.dpoi's info) in the DPoI modes.
if nargin < 4 || isempty(genie)
    genie = struct();
end
if nargin < 5
    options = struct();
end
trueChannel = any(strcmp(mode, {'knownH', 'dpoi_knownH'}));
if trueChannel
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
if trueChannel
    assert(isfield(genie, 'H'), '%s needs the true channel genie.H.', mode);
    H = genie.H;
    if isfield(genie, 'PAGain'), H = H .* genie.PAGain; end
end
switch mode
    case 'conventional'
        H = rx.interpolate(Hp, ctx);
        [xhat, nu] = rx.equalize(Y, H, sigma2, ctx);
    case 'knownH'
        [xhat, nu] = rx.equalize(Y, H, sigma2, ctx);
    case {'dpoi', 'smoothed', 'dpoi_genie', 'dpoi_knownH'}
        window = []; ridge = [];
        if isfield(options, 'DelayWindow'), window = options.DelayWindow; end
        if isfield(options, 'DelayRidge'), ridge = options.DelayRidge; end
        options = rmfield(options, intersect(fieldnames(options), {'DelayWindow', 'DelayRidge'}));
        if ~trueChannel
            [H, dof] = rx.delay_estimate(Hp, ctx, window, ridge);
            options.ChannelDof = dof * numel(ctx.PilotCols);
        end
        if strcmp(mode, 'smoothed')
            options.Fit = false;
        elseif strcmp(mode, 'dpoi_genie')
            assert(isfield(genie, 'Grid'), 'dpoi_genie needs the true grid genie.Grid.');
            options.Target = genie.Grid;
            options.ChannelDof = 0;      % Fitted on the whole grid, not the pilots.
        end
        if ~isfield(options, 'NoiseVariance'), options.NoiseVariance = sigma2; end
        [xhat, nu, trace.DPoI] = rx.dpoi(Y, H, ctx, options);
        sigma2 = trace.DPoI.Sigma2;
    otherwise
        error('Unknown receiver mode: %s.', mode);
end
llr = rx.demap_llr(xhat, nu, ctx.Modulation);

trace.Y = Y;
trace.Hp = Hp;
trace.H = H;
trace.Sigma2 = sigma2;
trace.Xhat = xhat;
trace.Nu = nu;
end
