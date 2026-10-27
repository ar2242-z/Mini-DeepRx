function waveform = ofdm_mod(carrier, grid, Nfft)
%OFDM_MOD IFFT and cyclic prefix: resource grid to time samples.
%   waveform = tx.ofdm_mod(carrier, grid, Nfft). Transmit windowing is
%   disabled to keep the waveform a plain CP-OFDM signal.
waveform = nrOFDMModulate(carrier, grid, 'Windowing', 0, 'Nfft', Nfft);
end
