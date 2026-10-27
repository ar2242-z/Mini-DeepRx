function passed = test_chan(referenceFile)
%% Check the channel (+chan)
% Run: test_chan
% 1. Against the earlier code (reference/ref_v0.mat), six slot cases:
%    timing offset, true H, noiseless received grid and interference grid
%    must match exactly. Noise uses its own stream in the new code, so only
%    its power is checked (variance N0 per resource element).
% 2. Physics:
%    a. static, short-delay channel without PA: received grid = H.*X;
%       with Doppler the error grows (intercarrier interference);
%    b. average channel power is one over many fading realizations;
%    c. measured SIR and SNR per resource element match the settings;
%    d. adding interference leaves the wanted signal unchanged.
% The FFT here is nrOFDMDemodulate on the slot window found with the
% genie timing offset, standing in for rx.sync and rx.ofdm_demod.

if nargin == 0
    referenceFile = fullfile(fileparts(mfilename('fullpath')),'reference','ref_v0.mat');
end
quiet = {'ShowPlots',false,'RunChecks',false,'SaveData',false,'Verbose',false};
passed = true;

%% 1. Channel against the legacy reference
fprintf('1. Channel versus legacy reference (%s)\n',referenceFile);
saved = load(referenceFile,'ref');
for c = fieldnames(saved.ref.slots).'
    s = saved.ref.slots.(c{1});
    cfg = fromStruct(s.cfg);
    if startsWith(cfg.ChannelModel,'CDL')
        % The legacy CDL used 45-degree slanted antennas (about half the
        % channel power); chan.fading uses vertical ones, so rays differ.
        fprintf('  SKIP  %-15s legacy CDL polarization differs by design (see 2b)\n',c{1});
        continue
    end
    [waveform,txOut] = tx.transmit(cfg,RandStream('mt19937ar','Seed',cfg.Seed));
    [~,ch] = chan.propagate(waveform,txOut.Carrier,cfg,RandStream('mt19937ar','Seed',1));
    demod = @(w) slotGrid(w,ch.TimingOffset,txOut.Carrier,cfg);
    noisePower = mean(abs(demod(ch.Noise)).^2,'all')/ch.NoiseVariance;
    errH = relErr(ch.H,s.chan.H);
    errY = relErr(demod(ch.Noiseless),s.chan.NoiselessY);
    errI = relErr(demod(ch.Interference),s.chan.Interference);
    ok = ch.TimingOffset==s.chan.TimingOffsetSamples && errH<1e-10 && ...
        errY<1e-10 && errI<1e-10 && abs(noisePower-1)<0.05;
    passed = report(ok,sprintf('%-15s H %.1e, noiseless Y %.1e, interference %.1e, noise power %.3f', ...
        c{1},errH,errY,errI,noisePower)) && passed;
end

%% 2a. Static channel gives Y = H.*X; Doppler adds intercarrier interference
fprintf('2a. Received grid versus H.*X (no noise, no PA)\n');
models = {'TDL-C',0;'CDL-D',0;'TDL-C',100;'TDL-C',500};
for k = 1:size(models,1)
    cfg = simo_config(quiet{:},'ChannelModel',models{k,1},'MaximumDopplerShift',models{k,2}, ...
        'DelaySpread',30e-9,'SNRdB',Inf);
    [waveform,txOut] = tx.transmit(cfg,RandStream('mt19937ar','Seed',5));
    [~,ch] = chan.propagate(waveform,txOut.Carrier,cfg,RandStream('mt19937ar','Seed',1));
    err = relErr(slotGrid(ch.Noiseless,ch.TimingOffset,txOut.Carrier,cfg),ch.H.*txOut.Grid);
    if models{k,2}==0
        ok = err < 1e-9; rule = '< 1e-9';
    else
        ok = err > 1e-9 && err < 0.1; rule = 'small but non-zero (ICI)';
    end
    passed = report(ok,sprintf('%s, Doppler %3d Hz: |Y-HX|/|Y| = %.2e (%s)', ...
        models{k,1},models{k,2},err,rule)) && passed;
end

%% 2b. Average channel power
fprintf('2b. Average channel power over 100 realizations\n');
for model = {'TDL-A','TDL-C','TDL-E','CDL-A','CDL-C','CDL-D'}
    cfg = simo_config(quiet{:},'ChannelModel',model{1},'SNRdB',Inf);
    [waveform,txOut] = tx.transmit(cfg,RandStream('mt19937ar','Seed',1));
    power = zeros(100,1);
    for seed = 1:100
        cfg.Seed = seed;
        [~,fade] = chan.fading(waveform,txOut.Carrier,cfg,seed);
        power(seed) = mean(abs(fade.H).^2,'all');
    end
    % Slot power fluctuates with fading; allow four standard errors.
    se = std(power)/sqrt(numel(power));
    passed = report(abs(mean(power)-1)<4*se,sprintf('%s: E|H|^2 = %.3f +- %.3f (expected 1)', ...
        model{1},mean(power),se)) && passed;
end

%% 2c. Measured SIR and SNR per occupied resource element
fprintf('2c. Measured SIR and SNR, averaged over 40 slots\n');
cfg = simo_config(quiet{:},'SIRdB',6,'SNRdB',10);
sig = 0; intf = 0; noise = 0;
noiseStream = RandStream('mt19937ar','Seed',3);
for seed = 1:40
    cfg.Seed = seed;
    [waveform,txOut] = tx.transmit(cfg,RandStream('mt19937ar','Seed',seed));
    [~,ch] = chan.propagate(waveform,txOut.Carrier,cfg,noiseStream);
    used = repmat(txOut.DataMask | txOut.PilotMask,1,1,cfg.NumReceiveAntennas);
    demod = @(w) slotGrid(w,ch.TimingOffset,txOut.Carrier,cfg);
    S = demod(ch.Noiseless); I = demod(ch.Interference); N = demod(ch.Noise);
    sig = sig + mean(abs(S(used)).^2);
    intf = intf + mean(abs(I(used)).^2);
    noise = noise + mean(abs(N(:)).^2);
end
sirdB = 10*log10(sig/intf); snrdB = 10*log10(sig/noise);
passed = report(abs(sirdB-cfg.SIRdB)<1,sprintf('SIR %.2f dB (set %g)',sirdB,cfg.SIRdB)) && passed;
passed = report(abs(snrdB-cfg.SNRdB)<1,sprintf('SNR %.2f dB (set %g)',snrdB,cfg.SNRdB)) && passed;

%% 2d. Interference does not change the wanted signal
cfg = simo_config(quiet{:},'SIRdB',Inf,'SNRdB',Inf);
[waveform,txOut] = tx.transmit(cfg,RandStream('mt19937ar','Seed',9));
[~,clean] = chan.propagate(waveform,txOut.Carrier,cfg,RandStream('mt19937ar','Seed',1));
cfg.SIRdB = 0;
[~,dirty] = chan.propagate(waveform,txOut.Carrier,cfg,RandStream('mt19937ar','Seed',1));
ok = isequal(clean.Noiseless,dirty.Noiseless) && isequal(clean.H,dirty.H) && ...
    any(dirty.Interference(:)) && ~any(clean.Interference(:));
passed = report(ok,'2d. Wanted signal and H identical with and without interference') && passed;

if passed, fprintf('All channel checks passed.\n'); else, fprintf('Some channel checks FAILED.\n'); end
end

%% Local helpers
function Y = slotGrid(waveform,offset,carrier,cfg)
    % Genie slot window, then FFT (stand-in for rx.sync + rx.ofdm_demod).
    ofdm = nrOFDMInfo(carrier,'Nfft',cfg.Nfft);
    n = carrier.SymbolsPerSlot;
    slotLength = sum(double(ofdm.CyclicPrefixLengths(1:n))) + n*cfg.Nfft;
    Y = nrOFDMDemodulate(carrier,waveform(offset+(1:slotLength),:),'Nfft',cfg.Nfft);
end

function e = relErr(a,b)
    e = norm(a(:)-b(:))/max(norm(b(:)),eps);
end

function cfg = fromStruct(saved)
    options = [fieldnames(saved).'; struct2cell(saved).'];
    cfg = simo_config(options{:});
end

function ok = report(ok,label)
    if ok, fprintf('  PASS  %s\n',label); else, fprintf('  FAIL  %s\n',label); end
end
