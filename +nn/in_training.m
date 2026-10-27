function tf = in_training(training, name)
%IN_TRAINING True if network name runs in training mode in a forward pass.
%   tf = nn.in_training(training, name)
%   training: true / false for every network, or a cell array of the names
%   of the networks being trained (nn.train passes its trainable list).
%   Frozen networks run in prediction mode, so their batch-norm statistics
%   stay as they were.
if iscell(training)
    tf = ismember(name, training);
else
    tf = logical(training);
end
end
