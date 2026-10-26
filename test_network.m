function [snrResults,dopplerResults,sirResults,paResults] = test_network(varargin)
%% Evaluate a saved CNN on ONE unseen seed, with SNR and Doppler sweeps
% Run after network has completed training: test_network
% Or choose another seed: test_network('TestSeed',9002)
% SNR: 0:2.5:20 dB at 100 Hz. Doppler: 0:100:500 Hz at 20 dB.
% Each point is ONE slot, not a long-run BER estimate.
% The modulation comes from the model's saved BaseConfig (16QAM by default).
% For a less noisy comparison: test_network('TestSeed',9001,'NumTrials',100)
% This uses seeds 9001:9100 and pools errors and bits at each sweep point.
% Both sweeps repeat for every channel model and pilot configuration. The
% defaults are those the model was trained on; test unseen ones with e.g.
%   test_network('ChannelModels',{'TDL-A','TDL-E'}, ...
%       'PilotConfigs',struct('DMRSAdditionalPosition',{0,1}))
% Interference is off by default (SIRdB Inf). Add an interferer to both
% sweeps, or an SIR sweep at 20 dB and 100 Hz, with e.g.
%   test_network('SIRdB',18,'SIRValues',0:6:36)
% The transmit PA is linear in the SNR, Doppler, and SIR sweeps. Add a PA
% input back-off sweep at 20 dB and 100 Hz (Rapp by default) with e.g.
%   test_network('PABackoffValues',0:2:10,'PAModel','polynomial')

p = inputParser; p.PartialMatching = false;
addParameter(p,'TestSeed',9001,@(x) isnumeric(x) && isscalar(x) && isreal(x) && ...
    isfinite(x) && x>=0 && x<=2^32-1 && x==fix(x));
addParameter(p,'ShowPlots',true,@(x) islogical(x) && isscalar(x));
addParameter(p,'NumTrials',1,@(x) isnumeric(x) && isscalar(x) && isreal(x) && ...
    isfinite(x) && x>=1 && x==fix(x));
addParameter(p,'ChannelModels',{},@(x) iscellstr(x) || isstring(x));
addParameter(p,'PilotConfigs',struct([]),@isstruct);
addParameter(p,'DelaySpread',[],@(x) isempty(x) || (isnumeric(x) && isscalar(x) && x>0));
addParameter(p,'SIRdB',Inf,@(x) isnumeric(x) && isscalar(x) && isreal(x) && (isfinite(x) || x==Inf));
addParameter(p,'SIRValues',[],@(x) isempty(x) || (isnumeric(x) && isvector(x) && isreal(x) && all(isfinite(x))));
addParameter(p,'PAModel','rapp',@(x) (ischar(x) || isstring(x)) && any(strcmp(x,{'polynomial','rapp'})));
addParameter(p,'PABackoffValues',[],@(x) isempty(x) || (isnumeric(x) && isvector(x) && isreal(x) && all(isfinite(x))));
parse(p,varargin{:}); options = p.Results;
options.PAModel = char(options.PAModel);
assert(~isempty(which('dlnetwork')), 'Testing the CNN requires Deep Learning Toolbox.');
folder = fileparts(mfilename('fullpath'));
modelFile = fullfile(folder,'trained_network.mat');
assert(isfile(modelFile),'Run network to train and save trained_network.mat first.');
saved = load(modelFile,'model'); model = saved.model;
testSeeds = options.TestSeed+(0:options.NumTrials-1);
assert(testSeeds(end)<=2^32-1,'Test seed range exceeds the generator limit.');
usedSeeds = [model.Training.TrainingSeeds(:);model.Training.ValidationSeeds(:)];
if isfield(model,'TrainingSamples')
    usedSeeds = [usedSeeds;model.TrainingSamples.Seed;model.ValidationSamples.Seed];
end
assert(isempty(intersect(testSeeds,usedSeeds)), ...
    'Test seeds must differ from all training and validation simulation/master seeds.');
cfg = model.BaseConfig;
assert(isfield(cfg,'Modulation'), ...
    'trained_network.mat predates the Modulation setting; retrain with network.');
cfg.Seed = options.TestSeed;
cfg.SNRdB = 20; cfg.MaximumDopplerShift = 100;
cfg.ShowPlots = false; cfg.RunChecks = false;
cfg.SaveData = false; cfg.Verbose = false;
cfg = slot_config(cfg,struct()); % Fill settings missing from older models.
if ~isempty(options.DelaySpread), cfg.DelaySpread = options.DelaySpread; end
cfg.SIRdB = options.SIRdB;
cfg.PAModel = 'none';
% Default to the conditions the model was trained on.
models = cellstr(options.ChannelModels);
if isempty(models)
    models = {cfg.ChannelModel};
    if isfield(model.Training,'ChannelModels'), models = model.Training.ChannelModels; end
end
pilotConfigs = options.PilotConfigs(:);
if isempty(pilotConfigs)
    pilotConfigs = struct('DMRSAdditionalPosition',cfg.DMRSAdditionalPosition);
    if isfield(model.Training,'PilotConfigs'), pilotConfigs = model.Training.PilotConfigs; end
end

%% Reuse identical received signals for all three receivers at each point
snrResults = table(); dopplerResults = table(); sirResults = table(); paResults = table();
for m = 1:numel(models)
    for k = 1:numel(pilotConfigs)
        caseCfg = slot_config(cfg,pilotConfigs(k));
        caseCfg.ChannelModel = models{m};
        fprintf('--- %s, %s ---\n',models{m},pilotLabel(caseCfg));
        snrResults = [snrResults; %#ok<AGROW>
            testSweep(model,caseCfg,'SNRdB',0:2.5:20,testSeeds)];
        dopplerResults = [dopplerResults; %#ok<AGROW>
            testSweep(model,caseCfg,'MaximumDopplerShift',0:100:500,testSeeds)];
        if ~isempty(options.SIRValues)
            sirResults = [sirResults; %#ok<AGROW>
                testSweep(model,caseCfg,'SIRdB',options.SIRValues,testSeeds)];
        end
        if ~isempty(options.PABackoffValues)
            paCfg = caseCfg;
            paCfg.PAModel = options.PAModel;
            paResults = [paResults; %#ok<AGROW>
                testSweep(model,paCfg,'PABackoffdB',options.PABackoffValues,testSeeds)];
        end
    end
end
save(fullfile(folder,'network_test_results.mat'),'snrResults','dopplerResults', ...
    'sirResults','paResults','options','cfg');
disp('SNR sweep (100 Hz):'); disp(snrResults);
disp('Doppler sweep (20 dB):'); disp(dopplerResults);
if ~isempty(sirResults), disp('SIR sweep (20 dB, 100 Hz):'); disp(sirResults); end
if ~isempty(paResults)
    fprintf('PA back-off sweep (%s, 20 dB, 100 Hz):\n',options.PAModel); disp(paResults);
end
fprintf('First test seed %d; checkpoint epoch %d; %d slots per point.\n', ...
    cfg.Seed,model.BestEpoch,options.NumTrials);
if options.ShowPlots
    seedLabel = sprintf('Test seed %d',cfg.Seed);
    if options.NumTrials>1
        seedLabel = sprintf('Test seeds %d to %d',testSeeds(1),testSeeds(end));
    end
    fig = plotTest(snrResults,'SNR (dB)',[seedLabel ': Doppler = 100 Hz']);
    exportgraphics(fig,fullfile(folder,'network_test_snr.png'),'Resolution',150);
    fig = plotTest(dopplerResults,'Maximum Doppler shift (Hz)', ...
        [seedLabel ': SNR = 20 dB']);
    exportgraphics(fig,fullfile(folder,'network_test_doppler.png'),'Resolution',150);
    if ~isempty(sirResults)
        fig = plotTest(sirResults,'SIR (dB)',[seedLabel ': SNR = 20 dB, Doppler = 100 Hz']);
        exportgraphics(fig,fullfile(folder,'network_test_sir.png'),'Resolution',150);
    end
    if ~isempty(paResults)
        fig = plotTest(paResults,'PA input back-off (dB)', ...
            sprintf('%s: %s PA, SNR = 20 dB, Doppler = 100 Hz',seedLabel,options.PAModel));
        exportgraphics(fig,fullfile(folder,'network_test_pa.png'),'Resolution',150);
    end
end
end

function result = testSweep(model,cfg,parameter,values,testSeeds)
    counts = zeros(numel(values),3); totalBits = zeros(numel(values),1);
    sigma2 = zeros(numel(values),1);
    for k = 1:numel(values)
        cfg.(parameter) = values(k);
        for seed = testSeeds
        cfg.Seed = seed;
        [data.X,data.H,data.Y,data.meta] = mini_simo_channel(cfg);
        [Z,labels,mask] = deeprx_sample(data.Y,data.X,data.meta);
        assert(isequal(size(Z),model.InputSize),'Test input size does not match the trained model.');
        % predict uses frozen batch-normalization statistics: no training on
        % this seed, no gradient updates, and no test labels passed to CNN.
        scores = gather(extractdata(predict(model.Net,dlarray(Z,'SSCB'))));
        assert(isequal(size(scores),size(labels)) && all(isfinite(scores(:))), ...
            'Unexpected CNN output shape or nonfinite scores.');
        cnnErrors = nnz((scores(mask)>0)~=labels(mask));
        [baseline,noiseEstimate] = decode_simo_ls(data);
        sigma2(k) = sigma2(k)+noiseEstimate/numel(testSeeds);
        counts(k,:) = counts(k,:)+[baseline.BitErrors.',cnnErrors];
        totalBits(k) = totalBits(k)+nnz(mask);
        end
        fprintf('%s=%g: known H %.4g, LS %.4g, CNN %.4g\n', ...
            parameter,values(k),counts(k,1)/totalBits(k), ...
            counts(k,2)/totalBits(k),counts(k,3)/totalBits(k));
    end
    result = table(values(:),counts(:,1),counts(:,2),counts(:,3),totalBits, ...
        counts(:,1)./totalBits,counts(:,2)./totalBits,counts(:,3)./totalBits,sigma2, ...
        'VariableNames',{'Value','ErrorsKnownH','ErrorsLS','ErrorsCNN','TotalBits', ...
        'BERKnownH','BERLS','BERCNN','EstimatedNoiseVariance'});
    result.ChannelModel = repmat(string(cfg.ChannelModel),height(result),1);
    result.Pilots = repmat(string(pilotLabel(cfg)),height(result),1);
    result = result(:,[end-1 end 1:end-2]);
end

function label = pilotLabel(cfg)
    numSymbols = (1+cfg.DMRSAdditionalPosition)*cfg.DMRSLength;
    label = sprintf('%d pilot symbol(s), type %d',numSymbols,cfg.DMRSConfigurationType);
end

function fig = plotTest(result,xLabel,plotTitle)
    % One tile per channel model and pilot configuration.
    [cases,~,group] = unique(result(:,{'ChannelModel','Pilots'}),'stable');
    fig = figure('Name',plotTitle);
    layout = tiledlayout(fig,'flow','TileSpacing','compact');
    title(layout,plotTitle);
    subtitle(layout,sprintf('%d data bits/point; triangles = zero errors, shown at 1 / bit count', ...
        result.TotalBits(1)));
    colors = lines(3); markers = {'o','s','d'};
    for c = 1:height(cases)
        rows = result(group==c,:);
        nexttile(layout); hold on;
        bers = [rows.BERKnownH,rows.BERLS,rows.BERCNN];
        for r = 1:3
            plotted = bers(:,r); zeroErrors = plotted==0;
            plotted(zeroErrors) = NaN;
            semilogy(rows.Value,plotted,['-' markers{r}], ...
                'Color',colors(r,:),'LineWidth',1.5);
            semilogy(rows.Value(zeroErrors),1./rows.TotalBits(zeroErrors),'v', ...
                'Color',colors(r,:),'HandleVisibility','off');
        end
        set(gca,'YScale','log'); grid on;
        xlabel(xLabel); ylabel('Uncoded BER');
        title(sprintf('%s, %s',cases.ChannelModel(c),cases.Pilots(c)));
        ylim([0.5/max(rows.TotalBits),1]);
        if c==1, legend('Known H','LS H','CNN','Location','best'); end
    end
end
