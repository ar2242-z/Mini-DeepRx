    %% BER sweeps: SNR, maximum Doppler shift, and PA back-off
% Run this file. It calls the existing generator and receiver functions.
% Both curves use pilot-estimated noise power, never the simulator variance.
% Two receive antennas, uncoded 16QAM (the Modulation default in
% simo_config). Both sweeps repeat for every channel model and pilot
% configuration listed in the settings below.

%% 1. Settings
snrValues = 0:2.5:20;          % dB; hold Doppler at 100 Hz.
dopplerValues = 0:100:500;     % Hz; hold SNR at 20 dB.
% PA input back-off in dB at 20 dB and 100 Hz; [] skips this sweep.
% The SNR and Doppler sweeps stay linear (PAModel 'none').
paBackoffValues = 0:2:10;
paModel = 'rapp';              % 'polynomial' or 'rapp'.
numTrials = 100;              % Independent one-slot trials at each point.
% 'TDL-A'...'TDL-E' and/or 'CDL-A'...'CDL-E'.
channelModels = {'TDL-C'};
% One struct element per pilot layout; missing fields keep their defaults.
pilotConfigs = struct('DMRSAdditionalPosition',{0});
% Increase numTrials for more reliable estimates of very small BERs.
baseCfg = simo_config('SNRdB',20,'MaximumDopplerShift',100, ...
    'ShowPlots',false,'RunChecks',false,'SaveData',false,'Verbose',false);
folder = fileparts(mfilename('fullpath'));

%% 2. Run the sweeps
% Each trial has a different seed. Reuse matching seeds across points so
% comparisons use matched bits/channel random draws where applicable.
% Known H is the linear channel: it does not include the PA's Bussgang gain,
% which LS absorbs into its pilot-based estimate.
snrResults = table(); dopplerResults = table(); paResults = table();
for m = 1:numel(channelModels)
    for k = 1:numel(pilotConfigs)
        caseCfg = slot_config(baseCfg,pilotConfigs(k));
        caseCfg.ChannelModel = channelModels{m};
        fprintf('--- %s, %s ---\n',channelModels{m},pilotLabel(caseCfg));
        snrResults = [snrResults; %#ok<AGROW>
            runSweep(caseCfg,'SNRdB',snrValues,numTrials)];
        dopplerResults = [dopplerResults; %#ok<AGROW>
            runSweep(caseCfg,'MaximumDopplerShift',dopplerValues,numTrials)];
        if ~isempty(paBackoffValues)
            paCfg = caseCfg;
            paCfg.PAModel = paModel;
            paResults = [paResults; %#ok<AGROW>
                runSweep(paCfg,'PABackoffdB',paBackoffValues,numTrials)];
        end
    end
end
disp('SNR sweep (Doppler = 100 Hz):'); disp(snrResults);
disp('Doppler sweep (SNR = 20 dB):'); disp(dopplerResults);
if ~isempty(paResults)
    fprintf('PA back-off sweep (%s, SNR = 20 dB, Doppler = 100 Hz):\n',paModel);
    disp(paResults);
end
save(fullfile(folder,'sweep_results.mat'), ...
    'snrResults','dopplerResults','paResults','paModel','baseCfg','numTrials', ...
    'channelModels','pilotConfigs');

%% 3. Two separate figures, also saved as PNG files
figSNR = plotSweep(snrResults,'SNR (dB)','BER versus SNR, Doppler = 100 Hz');
exportgraphics(figSNR,fullfile(folder,'sweep_snr.png'),'Resolution',150);
figDoppler = plotSweep(dopplerResults,'Maximum Doppler shift (Hz)', ...
    'BER versus Doppler, SNR = 20 dB');
exportgraphics(figDoppler,fullfile(folder,'sweep_doppler.png'),'Resolution',150);
fprintf('Saved sweep_results.mat, sweep_snr.png, and sweep_doppler.png.\n');
if ~isempty(paResults)
    figPA = plotSweep(paResults,'PA input back-off (dB)', ...
        sprintf('BER versus PA back-off (%s), SNR = 20 dB, Doppler = 100 Hz',paModel));
    exportgraphics(figPA,fullfile(folder,'sweep_pa.png'),'Resolution',150);
    fprintf('Saved sweep_pa.png.\n');
end

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
            [data.X,data.H,data.Y,data.meta] = mini_simo_channel(cfg);
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
    result.ChannelModel = repmat(string(baseCfg.ChannelModel),height(result),1);
    result.Pilots = repmat(string(pilotLabel(baseCfg)),height(result),1);
    result = result(:,[end-1 end 1:end-2]);
end

function label = pilotLabel(cfg)
    numSymbols = (1+cfg.DMRSAdditionalPosition)*cfg.DMRSLength;
    label = sprintf('%d pilot symbol(s), type %d',numSymbols,cfg.DMRSConfigurationType);
end

function fig = plotSweep(result,xLabel,plotTitle)
    % One tile per channel model and pilot configuration.
    [cases,~,group] = unique(result(:,{'ChannelModel','Pilots'}),'stable');
    fig = figure('Name',plotTitle);
    layout = tiledlayout(fig,'flow','TileSpacing','compact');
    title(layout,plotTitle);
    subtitle(layout,'Downward triangles: zero observed errors, shown at 1 / total bits');
    colors = lines(2);
    markers = {'o','s'};
    for c = 1:height(cases)
        rows = result(group==c,:);
        nexttile(layout); hold on;
        bers = [rows.BERKnownH rows.BERLS];
        for receiver = 1:2
            nonzeroBER = bers(:,receiver);
            zeroErrors = nonzeroBER==0;
            nonzeroBER(zeroErrors) = NaN; % A log axis cannot display zero.
            semilogy(rows.Value,nonzeroBER,['-' markers{receiver}], ...
                'Color',colors(receiver,:),'LineWidth',1.5);
            % Show zero-error runs as downward triangles at one error / bit count.
            % These markers are plotting limits, NOT measured nonzero BERs.
            semilogy(rows.Value(zeroErrors),1./rows.TotalBits(zeroErrors),'v', ...
                'Color',colors(receiver,:),'HandleVisibility','off');
        end
        set(gca,'YScale','log'); grid on;
        xlabel(xLabel); ylabel('Uncoded BER');
        title(sprintf('%s, %s',cases.ChannelModel(c),cases.Pilots(c)));
        ylim([0.5/max(rows.TotalBits),1]);
        if c==1, legend('Known H','LS H','Location','best'); end
    end
end
