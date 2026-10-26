
function cfg = simo_config(varargin)
%SIMO_CONFIG Create settings for the mini SIMO experiment.
%   cfg = simo_config() returns all default settings as a structure.
%   cfg = simo_config('SNRdB', 5, 'MaximumDopplerShift', 200)
%   changes only the named settings; the others keep their defaults.
%
%   You can also edit the returned structure:
%       cfg.SNRdB = 15;
%       [X,H,Y,meta] = mini_simo_channel(cfg);

    % inputParser pairs each parameter name with its value and checks it.
    p = inputParser;
    p.PartialMatching = false; % Catch misspelled names rather than guessing.
    positiveScalar = @(x) isnumeric(x) && isscalar(x) && isreal(x) && isfinite(x) && x>0;
    positiveInteger = @(x) positiveScalar(x) && x==fix(x);
    nonnegativeScalar = @(x) isnumeric(x) && isscalar(x) && isreal(x) && isfinite(x) && x>=0;
    logicalScalar = @(x) islogical(x) && isscalar(x);

    addParameter(p,'NumResourceBlocks',26,positiveInteger); % 12 subcarriers each.
    addParameter(p,'SubcarrierSpacing',15, ...
        @(x) isnumeric(x) && isscalar(x) && ismember(x,[15 30 60 120])); % kHz.
    addParameter(p,'Nfft',512,positiveInteger); % FFT size; room for channel filters.
    addParameter(p,'NumReceiveAntennas',2,positiveInteger);
    % 3GPP TR 38.901 channel: 'TDL-A'...'TDL-E' or 'CDL-A'...'CDL-E'.
    % D and E contain a line-of-sight component; A-C are non-line-of-sight.
    addParameter(p,'ChannelModel','TDL-C', ...
        @(x) (ischar(x) || isstring(x)) && any(strcmp(x,channelModels())));
    addParameter(p,'CarrierFrequency',3.5e9,positiveScalar); % Hz; CDL only.
    addParameter(p,'DelaySpread',300e-9,positiveScalar); % RMS delay spread, seconds.
    addParameter(p,'MaximumDopplerShift',100,nonnegativeScalar); % Hz; 0 is static.
    % Data modulation (5G TS 38.211 mapping); pilots are unaffected.
    addParameter(p,'Modulation','16QAM', ...
        @(x) (ischar(x) || isstring(x)) && any(strcmp(x,{'QPSK','16QAM'})));
    % PUSCH DMRS pilots (TS 38.211 6.4.1.1), single port, mapping type A.
    % Defaults give ONE pilot symbol at MATLAB column 3 on every 2nd subcarrier.
    addParameter(p,'DMRSTypeAPosition',2,@(x) isnumeric(x) && isscalar(x) && ismember(x,[2 3]));
    addParameter(p,'DMRSLength',1,@(x) isnumeric(x) && isscalar(x) && ismember(x,[1 2])); % 2 = double symbol.
    addParameter(p,'DMRSAdditionalPosition',0,@(x) isnumeric(x) && isscalar(x) && ismember(x,0:3));
    % Type 1: comb on every 2nd subcarrier. Type 2: pairs every 6 subcarriers.
    addParameter(p,'DMRSConfigurationType',1,@(x) isnumeric(x) && isscalar(x) && ismember(x,[1 2]));
    % CDM groups left free of data in pilot symbols; 1 puts data in the gaps.
    addParameter(p,'NumCDMGroupsWithoutData',1,@(x) isnumeric(x) && isscalar(x) && ismember(x,1:3));
    % Default operating point: 20 dB and 100 Hz. Inf means no added noise.
    addParameter(p,'SNRdB',10, ...
        @(x) isnumeric(x) && isscalar(x) && isreal(x) && (isfinite(x) || x==Inf));
    % Signal-to-interference ratio of one co-channel PUSCH interferer, in dB.
    % Inf (default) means no interferer and leaves every slot unchanged.
    addParameter(p,'SIRdB',Inf, ...
        @(x) isnumeric(x) && isscalar(x) && isreal(x) && (isfinite(x) || x==Inf));
    % Memoryless transmit power amplifier (PA) applied to the time waveform.
    % 'none' (default) is linear and leaves every slot unchanged.
    addParameter(p,'PAModel','none', ...
        @(x) (ischar(x) || isstring(x)) && any(strcmp(x,{'none','polynomial','rapp'})));
    % Input back-off: waveform RMS below the saturation amplitude, in dB.
    addParameter(p,'PABackoffdB',6, ...
        @(x) isnumeric(x) && isscalar(x) && isreal(x) && isfinite(x));
    addParameter(p,'RappSmoothness',2,positiveScalar); % Rapp p; larger is sharper.
    % Odd-order terms [a1 a3 a5 ...]: y = a1*u + a3*u|u|^2 + ... Input is held
    % at |u|<=1, so choose terms whose AM/AM curve rises up to |u|=1.
    % The default [1 -1/3] peaks exactly at |u|=1. Complex terms add AM/PM.
    addParameter(p,'PACoefficients',[1 -1/3], ...
        @(x) isnumeric(x) && isvector(x) && ~isempty(x) && all(isfinite(x)));
    addParameter(p,'Seed',42, ...
        @(x) nonnegativeScalar(x) && x==fix(x) && x<=2^32-1);
    addParameter(p,'ShowPlots',true,logicalScalar);
    addParameter(p,'RunChecks',true,logicalScalar);
    addParameter(p,'SaveData',true,logicalScalar); % Disable during sweeps.
    addParameter(p,'Verbose',true,logicalScalar);  % Print per-slot information.
    parse(p,varargin{:});
    cfg = p.Results;
    cfg.Modulation = char(cfg.Modulation);
    cfg.ChannelModel = char(cfg.ChannelModel);
    cfg.PAModel = char(cfg.PAModel);
    cfg.PACoefficients = cfg.PACoefficients(:).'; % Row vector.
    assert(cfg.NumResourceBlocks<=275,'NumResourceBlocks must not exceed 275.');
    assert(cfg.Nfft>12*cfg.NumResourceBlocks, ...
        'Nfft must exceed the number of active subcarriers (12*NumResourceBlocks).');
    assert(cfg.NumCDMGroupsWithoutData<=cfg.DMRSConfigurationType+1, ...
        'DMRS type 1 has 2 CDM groups; type 2 has 3.');
    assert(cfg.DMRSLength==1 || cfg.DMRSAdditionalPosition<=1, ...
        'Double-symbol DMRS allows at most one additional position.');
    assert(cfg.DMRSTypeAPosition==2 || cfg.DMRSAdditionalPosition<3, ...
        'DMRSAdditionalPosition 3 requires DMRSTypeAPosition 2.');
end

function names = channelModels()
    names = [strcat('TDL-',{'A','B','C','D','E'}),strcat('CDL-',{'A','B','C','D','E'})];
end
