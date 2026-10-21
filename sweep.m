%% BER sweeps: SNR and maximum Doppler shift
% Run this file. It calls the existing generator and receiver functions.
% Both curves use pilot-estimated noise power, never the simulator variance.
% One pilot-bearing OFDM symbol, two receive antennas, uncoded QPSK.

%% 1. Settings
snrValues = 0:2.5:20;          % dB; hold Doppler at 100 Hz.
dopplerValues = 0:100:500;     % Hz; hold SNR at 20 dB.
numTrials = 100;              % Independent one-slot trials at each point.
% Increase numTrials for more reliable estimates of very small BERs.
baseCfg = simo_config('SNRdB',20,'MaximumDopplerShift',100, ...
    'ShowPlots',false,'RunChecks',false,'SaveData',false,'Verbose',false);
folder = fileparts(mfilename('fullpath'));

%% 2. Run both sweeps
% Each trial has a different seed. Reuse matching seeds across points so
% comparisons use matched bits/channel random draws where applicable.
snrResults = runSweep(baseCfg,'SNRdB',snrValues,numTrials);
dopplerResults = runSweep(baseCfg,'MaximumDopplerShift',dopplerValues,numTrials);
disp('SNR sweep (Doppler = 100 Hz):'); disp(snrResults);
disp('Doppler sweep (SNR = 20 dB):'); disp(dopplerResults);
save(fullfile(folder,'sweep_results.mat'), ...
    'snrResults','dopplerResults','baseCfg','numTrials');

%% 3. Two separate figures, also saved as PNG files
figSNR = plotSweep(snrResults,'SNR (dB)','BER versus SNR, Doppler = 100 Hz');
exportgraphics(figSNR,fullfile(folder,'sweep_snr.png'),'Resolution',150);
figDoppler = plotSweep(dopplerResults,'Maximum Doppler shift (Hz)', ...
    'BER versus Doppler, SNR = 20 dB');
exportgraphics(figDoppler,fullfile(folder,'sweep_doppler.png'),'Resolution',150);
fprintf('Saved sweep_results.mat, sweep_snr.png, and sweep_doppler.png.\n');

%% Local functions
function result = runSweep(baseCfg,parameter,values,numTrials)
    counts = zeros(numel(values),2); % Columns: known H, LS H.
    totalBits = zeros(numel(values),1);
    meanSigma2 = zeros(numel(values),1);
    for k = 1:numel(values)
        cfg = baseCfg;
        cfg.(parameter) = values(k); % Change the field named by parameter.
        for trial = 1:numTrials
            cfg.Seed = baseCfg.Seed + trial - 1;
            [data.X,data.H,data.Y,data.meta] = mini_simo_tdl(cfg);
            [decoded,sigma2Hat] = decode_simo_ls(data);
            counts(k,:) = counts(k,:) + decoded.BitErrors.';
            totalBits(k) = totalBits(k) + decoded.TotalBits(1);
            meanSigma2(k) = meanSigma2(k) + sigma2Hat/numTrials;
        end
        fprintf('%s = %g: known H BER %.4g, LS H BER %.4g (%d bits)\n', ...
            parameter,values(k),counts(k,1)/totalBits(k), ...
            counts(k,2)/totalBits(k),totalBits(k));
    end
    % Divide pooled error counts by pooled bit counts, not individual BERs.
    result = table(values(:),counts(:,1),counts(:,2),totalBits, ...
        counts(:,1)./totalBits,counts(:,2)./totalBits,meanSigma2, ...
        'VariableNames',{'Value','ErrorsKnownH','ErrorsLS','TotalBits', ...
        'BERKnownH','BERLS','MeanEstimatedNoiseVariance'});
end

function fig = plotSweep(result,xLabel,plotTitle)
    fig = figure('Name',plotTitle);
    bers = [result.BERKnownH result.BERLS];
    colors = lines(2);
    markers = {'o','s'};
    hold on;
    for receiver = 1:2
        nonzeroBER = bers(:,receiver);
        zeroErrors = nonzeroBER==0;
        nonzeroBER(zeroErrors) = NaN; % A log axis cannot display zero.
        semilogy(result.Value,nonzeroBER,['-' markers{receiver}], ...
            'Color',colors(receiver,:),'LineWidth',1.5);
        % Show zero-error runs as downward triangles at one error / bit count.
        % These markers are plotting limits, NOT measured nonzero BERs.
        semilogy(result.Value(zeroErrors),1./result.TotalBits(zeroErrors),'v', ...
            'Color',colors(receiver,:),'HandleVisibility','off');
    end
    set(gca,'YScale','log'); grid on;
    xlabel(xLabel); ylabel('Uncoded BER'); title(plotTitle);
    legend('Known H','LS H, one pilot symbol','Location','best');
    subtitle('Downward triangles: zero observed errors, shown at 1 / total bits');
    ylim([0.5/max(result.TotalBits),1]);
end
