function [snrResults,dopplerResults] = test_network(varargin)
%% Evaluate a saved CNN on ONE unseen seed, with SNR and Doppler sweeps
% Run after network has completed training: test_network
% Or choose another seed: test_network('TestSeed',9002)
% SNR: 0:2.5:20 dB at 100 Hz. Doppler: 0:100:500 Hz at 20 dB.
% Each point is ONE slot (1944 data bits), not a long-run BER estimate.
% For a less noisy comparison: test_network('TestSeed',9001,'NumTrials',100)
% This uses seeds 9001:9100 and pools errors and bits at each sweep point.

p = inputParser; p.PartialMatching = false;
addParameter(p,'TestSeed',9001,@(x) isnumeric(x) && isscalar(x) && isreal(x) && ...
    isfinite(x) && x>=0 && x<=2^32-1 && x==fix(x));
addParameter(p,'ShowPlots',true,@(x) islogical(x) && isscalar(x));
addParameter(p,'NumTrials',1,@(x) isnumeric(x) && isscalar(x) && isreal(x) && ...
    isfinite(x) && x>=1 && x==fix(x));
parse(p,varargin{:}); options = p.Results;
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
cfg.Seed = options.TestSeed;
cfg.SNRdB = 20; cfg.MaximumDopplerShift = 100;
cfg.ShowPlots = false; cfg.RunChecks = false;
cfg.SaveData = false; cfg.Verbose = false;

%% Reuse identical received signals for all three receivers at each point
snrResults = testSweep(model,cfg,'SNRdB',0:2.5:20,testSeeds);
dopplerResults = testSweep(model,cfg,'MaximumDopplerShift',0:100:500,testSeeds);
save(fullfile(folder,'network_test_results.mat'),'snrResults','dopplerResults', ...
    'options','cfg');
disp('SNR sweep (100 Hz):'); disp(snrResults);
disp('Doppler sweep (20 dB):'); disp(dopplerResults);
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
end
end

function result = testSweep(model,cfg,parameter,values,testSeeds)
    counts = zeros(numel(values),3); totalBits = zeros(numel(values),1);
    sigma2 = zeros(numel(values),1);
    for k = 1:numel(values)
        cfg.(parameter) = values(k);
        for seed = testSeeds
        cfg.Seed = seed;
        [data.X,data.H,data.Y,data.meta] = mini_simo_tdl(cfg);
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
end

function fig = plotTest(result,xLabel,plotTitle)
    fig = figure('Name',plotTitle); hold on;
    bers = [result.BERKnownH,result.BERLS,result.BERCNN];
    colors = lines(3); markers = {'o','s','d'};
    for r = 1:3
        plotted = bers(:,r); zeroErrors = plotted==0;
        plotted(zeroErrors) = NaN;
        semilogy(result.Value,plotted,['-' markers{r}], ...
            'Color',colors(r,:),'LineWidth',1.5);
        semilogy(result.Value(zeroErrors),1./result.TotalBits(zeroErrors),'v', ...
            'Color',colors(r,:),'HandleVisibility','off');
    end
    set(gca,'YScale','log'); grid on;
    xlabel(xLabel); ylabel('Uncoded BER'); title(plotTitle);
    subtitle(sprintf('%d data bits/point; triangles = zero errors, shown at 1 / bit count',result.TotalBits(1)));
    legend('Known H','LS H, one pilot symbol','CNN','Location','best');
    ylim([0.5/max(result.TotalBits),1]);
end
