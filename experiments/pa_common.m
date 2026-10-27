function args = pa_common()
%PA_COMMON Shared settings for every receiver trained under PA distortion.
%   args = pa_common() returns name/value pairs for train_fcnn (and later
%   train_receiver). Every PA run uses these, so FCNN, TCNN and hybrids see
%   identical links, PAs and slots; a run adds only its own Name (and Flow):
%     addpath experiments
%     args = pa_common;
%     train_fcnn(args{:}, 'Name', 'fcnn_pa')
%   Link: 16QAM, LDPC rate 0.5, polynomial PA a1*u + a3*u|u|^2 (simo_config).
%   Each slot draws a back-off in [0 10] dB (0 dB: heavy distortion, 10 dB:
%   mild) and one of 30 dithered training PAs; validation uses 10 others.
%   No interferer, so the PA is the only new impairment.
%   Change a setting here and every PA experiment follows.
args = { ...
    'Link', {'Coding','ldpc','PAModel','polynomial','PACoefficients',[1 -1/3]}, ...
    'PABackoffRange', [0 10], ...
    'PADither', 0.1, 'NumTrainingPAs', 30, 'NumValidationPAs', 10, 'PASeed', 2023, ...
    'SIRRange', [Inf Inf]};
end
