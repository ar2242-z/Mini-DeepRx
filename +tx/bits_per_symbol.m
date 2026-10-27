function q = bits_per_symbol(modulation)
%BITS_PER_SYMBOL Bits per symbol: QPSK 2, 16QAM 4.
switch modulation
    case 'QPSK'
        q = 2;
    case '16QAM'
        q = 4;
    otherwise
        error('Unsupported modulation: %s.', modulation);
end
end
