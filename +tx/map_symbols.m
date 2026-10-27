function [symbols, bitsPerSymbol] = map_symbols(bits, modulation)
%MAP_SYMBOLS 5G symbol mapping (TS 38.211 5.1) with unit average power.
%   [symbols, bitsPerSymbol] = tx.map_symbols(bits, modulation) for
%   'QPSK' or '16QAM'. With Q = bitsPerSymbol, bits(Q*(k-1)+(1:Q)) label
%   symbols(k).
bitsPerSymbol = tx.bits_per_symbol(modulation);
assert(mod(numel(bits), bitsPerSymbol) == 0, ...
    'The number of bits must be a multiple of %d.', bitsPerSymbol);
symbols = nrSymbolModulate(bits, modulation);
end
