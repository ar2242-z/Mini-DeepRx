function A = transport_block_size(cfg, numDataREs)
%TRANSPORT_BLOCK_SIZE Information bits per slot for the target code rate.
%   A = tx.transport_block_size(cfg, numDataREs) follows TS 38.214 5.1.3.2
%   (nrTBS): about TargetCodeRate * bitsPerSymbol * numDataREs, rounded to
%   a size the LDPC segmentation supports. numDataREs counts the data
%   resource elements in the slot (DMRS excluded).
nPRB = cfg.NumResourceBlocks;
nrePerPRB = numDataREs/nPRB;
assert(nrePerPRB == fix(nrePerPRB), ...
    'Data resource elements must be equal in every resource block.');
A = double(nrTBS(cfg.Modulation, 1, nPRB, nrePerPRB, cfg.TargetCodeRate));
end
