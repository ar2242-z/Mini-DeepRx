function args = mppa_common()
%MPPA_COMMON Shared settings for receivers trained under the 3GPP MP PA.
%   args = mppa_common() returns name/value pairs for train_receiver (and
%   train_fcnn), like pa_common but with the memory polynomial PA of
%   MPPA.md and the numerology of Pihlajasalo et al. (IEEE TWC 2023,
%   Table II); a run adds only its own Name (and Flow):
%     addpath experiments
%     args = mppa_common;
%     train_receiver(args{:}, 'Flow', 'ftf', 'Name', 'ftf_mppa')
%   Link: 60 kHz, 1024-point FFT, 66 RB (792 subcarriers), 64QAM, LDPC rate
%   0.5, PA 'mp3gpp' (3GPP TR 38.803 31 GHz CMOS) run at 4x oversampling.
%   Channel: the default TDL-C conditions (unchanged).
%   Each slot draws an output back-off in [-1 7] dB relative to the PA's
%   1 dB compression point and one of 30 dithered training PAs; validation
%   uses 10 others. No interferer, so the PA is the only new impairment.
args = { ...
    'Link', {'Coding','ldpc','SubcarrierSpacing',60,'Nfft',1024,'NumResourceBlocks',66, ...
        'Modulation','64QAM','PAModel','mp3gpp','PAOversampling',4}, ...
    'PABackoffRange', [-1 7], ...
    'PADither', 0.05, 'NumTrainingPAs', 30, 'NumValidationPAs', 10, 'PASeed', 2023, ...
    'SIRRange', [Inf Inf]};
end
