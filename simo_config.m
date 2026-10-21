function cfg = simo_config(varargin)
%SIMO_CONFIG Create settings for the mini SIMO experiment.
%   cfg = simo_config() returns all default settings as a structure.
%   cfg = simo_config('SNRdB', 5, 'MaximumDopplerShift', 200)
%   changes only the named settings; the others keep their defaults.
%
%   You can also edit the returned structure:
%       cfg.SNRdB = 15;
%       [X,H,Y,meta] = mini_simo_tdl(cfg);

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
    addParameter(p,'DelaySpread',300e-9,positiveScalar); % RMS delay spread, seconds.
    addParameter(p,'MaximumDopplerShift',100,nonnegativeScalar); % Hz; 0 is static.
    % Default operating point: 20 dB and 100 Hz. Inf means no added noise.
    addParameter(p,'SNRdB',20, ...
        @(x) isnumeric(x) && isscalar(x) && isreal(x) && (isfinite(x) || x==Inf));
    addParameter(p,'Seed',42, ...
        @(x) nonnegativeScalar(x) && x==fix(x) && x<=2^32-1);
    addParameter(p,'ShowPlots',true,logicalScalar);
    addParameter(p,'RunChecks',true,logicalScalar);
    addParameter(p,'SaveData',true,logicalScalar); % Disable during sweeps.
    addParameter(p,'Verbose',true,logicalScalar);  % Print per-slot information.
    parse(p,varargin{:});
    cfg = p.Results;
    assert(cfg.NumResourceBlocks<=275,'NumResourceBlocks must not exceed 275.');
    assert(cfg.Nfft>12*cfg.NumResourceBlocks, ...
        'Nfft must exceed the number of active subcarriers (12*NumResourceBlocks).');
end
