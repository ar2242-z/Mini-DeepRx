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
%                     impairment. With Nt transmit antennas the data sees
%                     g.*H/sqrt(Nt) (tx.transmit's power scaling), so that
%                     is the channel used. With more than one layer the
%                     DMRS ports cannot be separated yet (rx.ls_estimate),
%                     so the noise power comes instead from the pilot
%                     residual with the true channel (pilotNoise below);
%                     detection is the per-RE MIMO LMMSE of rx.equalize.
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
%   rx.ldpc_decode when coded. 'knownH' handles any number of layers;
%   the other modes handle one layer (L = 1) for now: rx.ls_estimate
%   (DMRS port separation) and rx.dpoi reject more.
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
if trueChannel
    assert(isfield(genie, 'H'), '%s needs the true channel genie.H.', mode);
    H = genie.H;
    if isfield(genie, 'PAGain'), H = H .* genie.PAGain; end
    if ctx.NumTransmitAntennas > 1, H = H / sqrt(ctx.NumTransmitAntennas); end
end
if trueChannel && ctx.NumLayers > 1
    Hp = [];
    sigma2 = pilotNoise(Y, H, ctx);
else
    [Hp, sigma2] = rx.ls_estimate(Y, ctx);
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

function sigma2 = pilotNoise(Y, H, ctx)
    % Noise (plus interference and PA distortion) power per RE from the
    % pilots, given the true channel: the residual Y - sum_l H_l X_l at
    % the pilot REs, averaged over pilots, pilot symbols and antennas.
    % 1 x 1 x 1 x B. Used by 'knownH' with several layers, whose DMRS
    % ports share their REs and so defeat rx.ls_estimate's estimate.
    B = size(Y, 4);
    Yp = Y(ctx.PilotRows, ctx.PilotCols, :, :);
    Hp = H(ctx.PilotRows, ctx.PilotCols, :, :, :);
    r = Yp - sum(Hp .* ctx.PilotValues, 5);
    r = reshape(r, [], B);
    sigma2 = reshape(mean(real(r).^2 + imag(r).^2, 1), 1, 1, 1, B);
end
