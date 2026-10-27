function args = pa_schedule()
%PA_SCHEDULE PA experiment with a scheduled learning rate and batch 10.
%   args = pa_schedule() returns pa_common's link, PAs and conditions plus:
%     MiniBatchSize 10   100 Adam steps per epoch (1000 slots), 15 000 in all
%     LearningRate 3e-3  peak; WarmupSteps 200 (2 epochs) from 0 to it,
%                        then linear decay to FinalLearningRate 2e-3 at the
%                        last step (DecayStart 0: decay right after warmup)
%     GradientClip 20    caps each network's gradient norm per step; only
%                        spikes are cut (measured at the start of training,
%                        batch 10: FCNN about 2, TCNN 6-10 with spikes to
%                        250, calibration 3-18 with spikes to 180). The
%                        epoch log and history.GradientNorm show the median
%                        norms; keep the clip at 3-5x the TCNN's.
%   Train every receiver compared with these, so they share one recipe:
%     addpath experiments
%     args = pa_schedule;
%     train_receiver(args{:}, 'Flow','tcnn_feq', 'Name','tcnn_feq_s', 'FindLearningRate',true)
%     train_receiver(args{:}, 'Flow','tcnn_feq', 'Name','tcnn_feq_s')
%     train_receiver(args{:}, 'Flow','fcnn', 'Name','fcnn_pa_s')
%   Continue an interrupted run with the same call plus 'Resume',true; start
%   from a saved model with 'InitFrom','models/<name>.mat'. A later
%   name/value pair overrides these, e.g. 'LearningRate',2e-3.
args = [pa_common(), { ...
    'MiniBatchSize', 10, ...
    'LearningRate', 5e-3, 'LearningRateSchedule', 'linear', ...
    'WarmupSteps', 200, 'DecayStart', 0, 'FinalLearningRate', 2e-3, ...
    'GradientClip', 20}];
end
