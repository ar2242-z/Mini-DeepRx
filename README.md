# Mini DeepRx simulation in MATLAB

A SIMO OFDM experiment with one transmit antenna, two receive antennas, a TDL-C channel, QPSK, and one pilot-bearing OFDM symbol. It compares a known-channel receiver, an LS-based receiver, and a small DeepRx-style CNN. BER is uncoded; LDPC is not implemented.

Run commands from this folder in MATLAB. Signal generation needs 5G Toolbox; CNN training and inference also need Deep Learning Toolbox. GPU execution uses Parallel Computing Toolbox when available.

| MATLAB file | Purpose | How to use it |
|---|---|---|
| `simo_config.m` | Creates and validates simulation settings: PRBs, subcarrier spacing, SNR, Doppler, antennas, delay spread, and random seed. | `cfg = simo_config('SNRdB',10);` |
| `mini_simo_tdl.m` | Generates data and pilots, OFDM-modulates them, applies TDL-C, demodulates, and adds noise. Returns `X`, true `H`, received `Y`, and supporting information in `meta`. Includes checks and channel/grid plots. | `[X,H,Y,meta] = mini_simo_tdl(cfg);` |
| `decode_simo_ls.m` | Compares known H against LS estimation from one pilot-bearing symbol. Estimates noise power from pilots, performs LMMSE antenna combining/equalization, demodulates QPSK, and calculates BER. Plots channels, constellations, and BER when run without inputs. | `decode_simo_ls;` after generating a saved slot. |
| `sweep.m` | Generates independent trials and plots known-H and LS BER versus SNR and Doppler. Uses 100 trials per point by default. | `sweep;` |
| `deeprx_sample.m` | Shared helper that assembles real-valued CNN input `Z`, target bit grids, and the mask excluding pilots from the loss. Used by both training and testing. | Normally called automatically. |
| `network.m` | Generates training/validation data, builds a five-block residual CNN, trains with masked binary cross-entropy and Adam, saves the best validation checkpoint, and plots training history. | `network;` |
| `test_network.m` | Loads the trained CNN and compares CNN, LS, and known-H BER on unseen seeds across both sweeps. Saves results and plots. | `test_network('TestSeed',9001);` |

**Typical run order**

```matlab
% Generate one slot and inspect the conventional receivers.
cfg = simo_config();
[X,H,Y,meta] = mini_simo_tdl(cfg);
decode_simo_ls;

% Conventional receiver sweeps; generates its own slots.
sweep;

% CNN training; generates its own training and validation slots.
network('TrainingSeeds',42:46,'SamplesPerSeed',100,'MaxEpochs',20);

% CNN evaluation: one unseen slot per sweep point.
test_network('TestSeed',9001);

% Optional: pool 20 unseen seeds for a less noisy BER estimate.
test_network('TestSeed',9001,'NumTrials',20);
```

**Settings and dimensions**

The current defaults in `simo_config.m` are **26 PRBs**, **15 kHz** subcarrier spacing, **20 dB** SNR, **100 Hz** maximum Doppler, **300 ns** RMS delay spread, FFT size **512**, and seed **42**. Each PRB contains 12 subcarriers, so the current grid has 312 subcarriers and 14 OFDM symbols.

Arrays use toolbox order: **frequency, time, antenna/feature**:

- `X`: `312 × 14`; transmitted data and pilots.
- `H`, `Y`: `312 × 14 × 2`; true channel and received signals.
- `Z`: `312 × 14 × 10` per sample; real and imaginary parts of `[Y, Xp, Hraw]`, where `Xp` is the pilot-only grid and `Hraw = Y .* conj(Xp)`.
- Bit labels and masks: `312 × 14 × 2`; two QPSK bits per resource element.
- Training batches add a fourth dimension for samples.

Change defaults in `simo_config.m`, or override settings for a single generation call:

```matlab
cfg = simo_config('NumResourceBlocks',12,'SubcarrierSpacing',30, ...
    'SNRdB',10,'MaximumDopplerShift',200,'Seed',123);
[X,H,Y,meta] = mini_simo_tdl(cfg);
```

`Nfft` must exceed `12 * NumResourceBlocks`. A local `cfg` override does not change settings inside `network` or `sweep`; those functions create their own configuration. Change shared defaults before retraining if you want a different grid. `test_network` uses the configuration saved with the model.

Some older comments and saved results refer to the earlier 6-PRB, 72-subcarrier experiment. Regenerate data and retrain after changing the grid or waveform settings.

**Training and sweep controls**

`TrainingSeeds` are master seeds for reproducible random streams, not five repeatedly reused slots. The default five master seeds generate 100 fresh slots each per epoch: **500 slots per epoch**, or **10,000 distinct slots over 20 epochs**. Validation uses two separate master seeds and 50 slots each. The best model is selected by validation loss, not test BER.

```matlab
network('TrainingSeeds',42:51,'SamplesPerSeed',100,'MaxEpochs',40);
network('PrepareOnly',true); % Generate Z and labels without training.
```

Edit `snrValues`, `dopplerValues`, `numTrials`, and `baseCfg` at the top of `sweep.m` for conventional sweeps. Both sweep files currently use SNR `0:2.5:20` dB at 100 Hz, and Doppler `0:100:500` Hz at 20 dB. CNN testing defaults to one trial; `NumTrials` adds consecutive test seeds and pools errors and bit counts.

**Saved outputs**

| Output | Contents |
|---|---|
| `mini_simo_tdl_data.mat` | One generated slot and its configuration. |
| `decode_simo_ls_results.mat` | LS channel estimate, estimated noise power, equalized symbols, detected bits, and BER. |
| `sweep_results.mat`, `sweep_*.png` | Conventional BER sweeps and plots. |
| `network_data.mat` | First-epoch inputs/labels/masks, fixed validation data, and the seed plan for all epochs. |
| `trained_network.mat`, `network_training.png` | Best CNN checkpoint, training settings/history, and learning curves. |
| `network_test_results.mat`, `network_test_*.png` | CNN and baseline test comparisons. |

Reruns replace their corresponding outputs. Files in `cnn_audit_before` and `cnn_audit_after_single` are historical snapshots, not the current implementation; do not add those folders to the MATLAB path.

The CNN is a learning experiment, not a full reproduction of the paper. It has not yet outperformed LS in the recorded aggregate tests. Zero observed errors in a short test do not establish zero long-run BER.
