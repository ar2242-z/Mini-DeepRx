# Mini DeepRx simulation in MATLAB

A SIMO OFDM experiment with one transmit antenna, two receive antennas, a 3GPP TDL or CDL channel (A–E, TDL-C by default), 16QAM data (QPSK selectable), and configurable 5G DMRS pilots (one pilot-bearing OFDM symbol by default). It compares a known-channel receiver, an LS-based receiver, and a small DeepRx-style CNN. BER is uncoded; LDPC is not implemented.

Run commands from this folder in MATLAB. Signal generation needs 5G Toolbox; CNN training and inference also need Deep Learning Toolbox. GPU execution uses Parallel Computing Toolbox when available.

| MATLAB file | Purpose | How to use it |
|---|---|---|
| `simo_config.m` | Creates and validates simulation settings: PRBs, subcarrier spacing, SNR, Doppler, antennas, delay spread, and random seed. | `cfg = simo_config('SNRdB',10);` |
| `mini_simo_channel.m` | Generates data and pilots, OFDM-modulates them, applies the selected TDL/CDL channel, demodulates, and adds noise. Returns `X`, true `H`, received `Y`, and supporting information in `meta`. Includes checks and channel/grid plots. | `[X,H,Y,meta] = mini_simo_channel(cfg);` |
| `decode_simo_ls.m` | Compares known H against LS estimation from the pilots (linear in frequency, then linear in time between pilot symbols). Estimates noise power from pilots, performs LMMSE antenna combining/equalization (bias-removed for 16QAM amplitude decisions), hard-demaps 16QAM, and calculates BER. Plots channels, constellations, and BER when run without inputs. | `decode_simo_ls;` after generating a saved slot. |
| `sweep.m` | Generates independent trials and plots known-H and LS BER versus SNR and Doppler. Uses 100 trials per point by default. | `sweep;` |
| `deeprx_sample.m` | Shared helper that assembles real-valued CNN input `Z`, target bit grids, and the mask excluding pilots from the loss. Used by both training and testing. | Normally called automatically. |
| `network.m` | Generates training/validation data, builds the 11-block residual CNN from the paper's Table I, trains with masked binary cross-entropy and Adam, saves the best validation checkpoint, and plots training history. | `network;` |
| `test_network.m` | Loads the trained CNN and compares CNN, LS, and known-H BER on unseen seeds across both sweeps. Saves results and plots. | `test_network('TestSeed',9001);` |
| `TCNN.m` | TCNN/FEQ (Pihlajasalo et al. 2023, Sec. III-C): a 5-block residual CNN on the received **time** samples, followed by fixed, differentiable FFT, LS/LMMSE and max-log demapping layers identical to `decode_simo_ls.m`. Trains end to end on bits with dithered PAs (30 training, 10 unseen validation PAs) and random back-off. Epoch 0 is the plain LS receiver; also reports validation BER versus back-off. | `TCNN;` |

**Typical run order**

```matlab
% Generate one slot and inspect the conventional receivers.
cfg = simo_config();
[X,H,Y,meta] = mini_simo_channel(cfg);
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

The current defaults in `simo_config.m` are **16QAM** data, **26 PRBs**, **15 kHz** subcarrier spacing, **20 dB** SNR, **100 Hz** maximum Doppler, **300 ns** RMS delay spread, FFT size **512**, and seed **42**. Each PRB contains 12 subcarriers, so the current grid has 312 subcarriers and 14 OFDM symbols.

Arrays use toolbox order: **frequency, time, antenna/feature**:

- `X`: `312 × 14`; transmitted data and pilots.
- `H`, `Y`: `312 × 14 × 2`; true channel and received signals.
- `Z`: `312 × 14 × 10` per sample; real and imaginary parts of `[Y, Xp, Hraw]`, where `Xp` is the pilot-only grid and `Hraw = Y .* conj(Xp)`.
- Bit labels and masks: `312 × 14 × 4`; four 16QAM bits per resource element (quadrant bits first, as in the paper). QPSK gives `312 × 14 × 2`.
- Training batches add a fourth dimension for samples.
- `meta.TimeDomainY`: `552 × 14 × 2`; the same received signal before the FFT, one column per OFDM symbol with its cyclic prefix, zero-padded so the FFT window is rows 21–532 in every column. `meta.TimeGrid` holds the window rows, active FFT bins, and phase correction that map it exactly to `Y`. Noise and interference are added to these time samples (variance `10^(-SNRdB/10)/Nfft` per sample, so each subcarrier keeps the same SNR); results are statistically unchanged, but slot-by-slot noise draws differ from runs before this change.

Change defaults in `simo_config.m`, or override settings for a single generation call:

```matlab
cfg = simo_config('NumResourceBlocks',12,'SubcarrierSpacing',30, ...
    'SNRdB',10,'MaximumDopplerShift',200,'Seed',123);
[X,H,Y,meta] = mini_simo_channel(cfg);
```

`Nfft` must exceed `12 * NumResourceBlocks`. A local `cfg` override does not change settings inside `network` or `sweep`; those functions create their own configuration. Change shared defaults before retraining if you want a different grid. `test_network` uses the configuration saved with the model.

Some older comments and saved results refer to the earlier 6-PRB, 72-subcarrier experiment. Regenerate data and retrain after changing the grid, waveform, or modulation settings; slots and models saved before the `Modulation` setting existed (QPSK) are rejected. Use `simo_config('Modulation','QPSK')` for the old behaviour.

**Channel models and pilots**

`simo_config` selects the channel with `ChannelModel` (`'TDL-A'`…`'TDL-E'`, `'CDL-A'`…`'CDL-E'`; D and E include line of sight). CDL uses one isotropic transmit element, a 1×`NumReceiveAntennas` isotropic receive array, and `CarrierFrequency` (3.5 GHz). Pilots follow 5G PUSCH DMRS: `DMRSTypeAPosition` (2 or 3), `DMRSLength` (1 or 2), `DMRSAdditionalPosition` (0–3 extra pilot symbols), `DMRSConfigurationType` (1: every 2nd subcarrier; 2: pairs every 6), and `NumCDMGroupsWithoutData`.

```matlab
cfg = simo_config('ChannelModel','CDL-D','DMRSAdditionalPosition',1); % pilots on symbols 3 and 12
[X,H,Y,meta] = mini_simo_channel(cfg);
```

Training can mix conditions; each slot draws one model and one pilot configuration, plus SNR, maximum Doppler shift, SIR, and RMS delay spread, each independently uniform over `SNRRange` (default −4 to 32 dB), `DopplerRange` (0–500 Hz), `SIRRange` (0–36 dB), and `DelaySpreadRange` (10–300 ns). SIR adds one co-channel PUSCH interferer (`simo_config` `SIRdB`, default `Inf` = none) with its own bits, DMRS scrambling, and fading; `'SIRRange',[Inf Inf]` trains without it. Testing defaults to the trained conditions (without interference) and accepts unseen ones:

```matlab
network('ChannelModels',{'CDL-A','CDL-B','CDL-C','CDL-D','CDL-E'}, ...
    'DelaySpreadRange',[10e-9 300e-9], ...
    'PilotConfigs',struct('DMRSAdditionalPosition',{0,1}));
test_network('ChannelModels',{'TDL-A','TDL-E'}, ...
    'PilotConfigs',struct('DMRSAdditionalPosition',{0,1}));
test_network('SIRValues',0:6:36); % Adds an SIR sweep at 20 dB and 100 Hz.
```

`sweep.m` has `channelModels` and `pilotConfigs` lists at the top. Results tables gain `ChannelModel` and `Pilots` columns, and plots show one tile per combination.

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
| `mini_simo_channel_data.mat` | One generated slot and its configuration. |
| `decode_simo_ls_results.mat` | LS channel estimate, estimated noise power, equalized symbols, detected bits, and BER. |
| `sweep_results.mat`, `sweep_*.png` | Conventional BER sweeps and plots. |
| `network_data.mat` | First-epoch inputs/labels/masks, fixed validation data, and the seed plan for all epochs. |
| `trained_network.mat`, `network_training.png` | Best CNN checkpoint, training settings/history, and learning curves. |
| `tcnn_data.mat` | TCNN first-epoch time-domain inputs, validation data, PA banks, and receiver operators. About 150 MB at defaults, so it is git-ignored. |
| `trained_tcnn.mat`, `tcnn_training.png`, `tcnn_backoff.png` | Best TCNN checkpoint with its fixed receiver, PA banks, history, and validation BER versus back-off (LS versus TCNN/FEQ). |
| `network_test_results.mat`, `network_test_*.png` | CNN and baseline test comparisons. |

Reruns replace their corresponding outputs. Files in `cnn_audit_before` and `cnn_audit_after_single` are historical snapshots, not the current implementation; do not add those folders to the MATLAB path.

The CNN is a learning experiment, not a full reproduction of the paper. It has not yet outperformed LS in the recorded aggregate tests. Zero observed errors in a short test do not establish zero long-run BER.
