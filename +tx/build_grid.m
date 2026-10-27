function [grid, dataMask, pilotMask] = build_grid(carrier, res, symbols)
%BUILD_GRID Place data symbols and DMRS in the resource grid.
%   [grid, dataMask, pilotMask] = tx.build_grid(carrier, res, symbols)
%   with res from tx.resources. Rows are subcarriers, columns OFDM symbols.
%   A mask is a logical array: true marks a resource element of that type.
assert(numel(symbols) == numel(res.DataIndices), ...
    'Got %d data symbols for %d data resource elements.', ...
    numel(symbols), numel(res.DataIndices));
grid = nrResourceGrid(carrier, 1);
grid(res.DataIndices) = symbols;
grid(res.PilotIndices) = res.Pilots;
dataMask = false(size(grid));
pilotMask = false(size(grid));
dataMask(res.DataIndices) = true;
pilotMask(res.PilotIndices) = true;
end
