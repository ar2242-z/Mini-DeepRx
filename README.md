# Mini DeepRx: link-level simulation in MATLAB

A single-input, multiple-output (SIMO) 5G uplink link — one transmit antenna, two
receive antennas — built to compare a conventional receiver with neural receivers
(TCNN, FCNN/DeepRx, and hybrids) on identical signals.

The code is organised along the signal chain. Each physical processing block is
one function, grouped by where it sits in the chain:

```
+tx/    transmitter        bits → LDPC → symbol mapping → resource grid → IFFT + CP → PA
+chan/  channel            fading → interference → noise
+rx/    receiver           sync → FFT → channel estimation → equalization → LLRs → LDPC decoding
+nn/    neural receivers   FCNN (DeepRx) and training; TCNN and hybrids next
```

All blocks share one configuration struct from `simo_config.m`. The rebuild plan is in
[remake.md](remake.md). The earlier scripts (`mini_simo_channel.m`, `decode_simo_ls.m`,
`network.m`, `TCNN.m`, `sweep.m`, …) were removed; they remain in git history (commit
`b29a596`). `reference/` keeps what the tests compare against: `ref_v0.mat` (fixed-seed
outputs of the earlier code) and `legacy_fcnn.mat` (the earlier trained DeepRx network).

Requirements: MATLAB with 5G Toolbox. Neural receivers will also need Deep Learning Toolbox.

## Transmitter (`+tx`)

```
          ┌──────┐  ┌──────┐  ┌───────────┐  ┌──────────┐  ┌──────────┐  ┌────┐
 source → │ bits │→ │ LDPC │→ │ map       │→ │ resource │→ │ IFFT +CP │→ │ PA │→ waveform
          └──────┘  └──────┘  │ symbols   │  │ grid     │  └──────────┘  └────┘
                    optional  └───────────┘  └────▲─────┘
                                                  │ DMRS pilots (resources)
```

One call runs the whole chain for one slot:

```matlab
cfg = simo_config('Coding','ldpc');                 % or 'none' for uncoded bits
stream = RandStream('mt19937ar','Seed',cfg.Seed);
[waveform, txOut] = tx.transmit(cfg, stream);
```

`waveform` is the PA output. `txOut` keeps every intermediate result, so each stage can be
inspected: `InfoBits`, `CodedBits`, `Code`, `Symbols`, `Grid`, `DataMask`, `PilotMask`,
`WaveformBeforePA`, `PAGain`, `Carrier`, `Resources`.

### Blocks, in signal order

The numbers below are for the default settings: 26 resource blocks, 15 kHz subcarrier
spacing, FFT size 512, 16QAM, one DMRS symbol, code rate 0.5.

| Function | Physical block | What it does |
|---|---|---|
| `tx.carrier` | Air interface (numerology) | Sets the time–frequency grid: 26 RB × 12 = **312 subcarriers** at 15 kHz (≈ 4.7 MHz), and a **1 ms slot of 14 OFDM symbols**. The cell ID scrambles the pilot sequence, so a second cell (the interferer, ID 1) has different pilots. No signal processing. |
| `tx.resources` | Resource allocation + reference-signal generation | Decides which resource elements (one subcarrier × one symbol) carry data and which carry **DMRS pilots**, and generates the pilot values (a scrambled QPSK sequence the receiver knows). The pilots let the receiver measure the channel; their spacing in frequency and time sets how much delay spread and Doppler it can follow. Default: 4212 data REs, 156 pilot REs. |
| `tx.bits` | Information source | Random 0/1 payload bits (the transport block). |
| `tx.transport_block_size` | Transport block sizing | How many information bits fit in one slot at the target code rate (5G rule, TS 38.214). Default: **8064 bits**. |
| `tx.ldpc_encode` | Channel encoder (5G UL-SCH, TS 38.212) | Adds redundancy so the receiver can correct errors: **CRC** (lets the receiver check a decoded block) → **segmentation** into LDPC-sized code blocks → **LDPC encoding** (parity bits) → **rate matching** (puncture/repeat and interleave so exactly G = 4 × 4212 = **16848 bits** fill the data REs). Effective rate 0.48. Skipped when `Coding` is `'none'`. |
| `tx.map_symbols` | Constellation mapper (digital modulator) | Each group of Q bits (2 for QPSK, 4 for 16QAM) becomes one complex value — the **amplitude and phase** one subcarrier carries for one symbol. 5G Gray mapping, unit average energy, so SNR per resource element is well defined. |
| `tx.build_grid` | Resource-element mapper | Places data symbols and pilots in the **312 × 14 time–frequency grid**, the frequency-domain picture of the slot. Also returns data and pilot masks. |
| `tx.ofdm_mod` | OFDM modulator | **IFFT** turns each grid column into 512 time samples (sample rate 512 × 15 kHz = 7.68 MHz; unused bins form the guard band). The **cyclic prefix** (36–40 samples, ≈ 4.7 µs) copies the end of each symbol to its front, so multipath shorter than the CP looks like a per-subcarrier gain, **Y = H·X**, after the receiver's FFT. Output: 7680 complex baseband samples per slot. |
| `tx.pa` | RF power amplifier | Memoryless non-linear amplifier (Rapp, or a polynomial with optional AM/PM). OFDM has a high peak-to-average power ratio, so peaks are compressed; **back-off** (`PABackoffdB`) sets the average level below saturation. The output is rescaled to the input power, so only distortion is added. `PAGain` is the Bussgang gain (output = gain × input + uncorrelated distortion). `PAModel` `'none'` is a linear PA. |
| `tx.transmit` | Whole transmitter | Runs the blocks above in order and returns the waveform and every intermediate result. |
| `tx.bits_per_symbol` | Helper | QPSK → 2, 16QAM → 4. |

Not modelled: data scrambling (TS 38.211 6.3.1.1), transmit windowing, and
up-conversion to the carrier frequency (all signals are complex baseband).

### Transmitter settings (`simo_config`)

| Setting | Default | Meaning |
|---|---|---|
| `NumResourceBlocks` | 26 | Bandwidth, 12 subcarriers per block |
| `SubcarrierSpacing` | 15 | kHz |
| `Nfft` | 512 | FFT size; must exceed 12 × `NumResourceBlocks` |
| `Modulation` | `'16QAM'` | `'QPSK'` or `'16QAM'` |
| `Coding` | `'none'` | `'ldpc'` for 5G UL-SCH coding |
| `TargetCodeRate` | 0.5 | Information bits / coded bits (LDPC only) |
| `DMRSTypeAPosition`, `DMRSLength`, `DMRSAdditionalPosition`, `DMRSConfigurationType`, `NumCDMGroupsWithoutData` | 2, 1, 0, 1, 1 | DMRS pilot pattern (TS 38.211 6.4.1.1); default is one pilot symbol at MATLAB column 3 on every 2nd subcarrier |
| `PAModel` | `'none'` | `'none'`, `'rapp'` or `'polynomial'` |
| `PABackoffdB` | 6 | Average power below PA saturation, dB |
| `RappSmoothness`, `PACoefficients` | 2, `[1 -1/3]` | PA model parameters |
| `Seed` | 42 | Random seed for the slot |

```matlab
cfg = simo_config('Modulation','QPSK','Coding','ldpc','TargetCodeRate',0.3, ...
    'PAModel','rapp','PABackoffdB',3);
```

## Channel (`+chan`)

```
                 ┌───────────────┐        ┌───┐        ┌───┐
 Tx waveform  →  │ fading        │──────→ │ + │──────→ │ + │ → received samples (per Rx antenna)
 (1 antenna)     │ TDL/CDL, Nr=2 │        └─▲─┘        └─▲─┘
                 └───────────────┘          │            │
                     interferer: own Tx + own fading     receiver noise (AWGN)
```

One call runs the whole channel; noise has its own random stream:

```matlab
[rxWaveform, chOut] = chan.propagate(waveform, txOut.Carrier, cfg, ...
    RandStream('mt19937ar','Seed',1));
```

`rxWaveform` is samples × receive antennas. It is longer than the transmitted slot by
`chOut.Padding` samples, because the channel delays the signal; locating the slot is the
receiver's job. `chOut` keeps the three parts of the sum (`Noiseless`, `Interference`,
`Noise`) and the **truth** used only for evaluation and the known-channel baseline: the
channel response `H` (312 × 14 × 2) and the perfect `TimingOffset`. A practical receiver
must not read them.

| Function | Physical block | What it does |
|---|---|---|
| `chan.fading` | Multipath radio channel (3GPP TR 38.901) | The signal reaches each receive antenna over many delayed, attenuated paths. Different delays make the channel **frequency selective** (`DelaySpread`, default 300 ns); motion gives each path a **Doppler** shift, so the channel **changes over the slot** (`MaximumDopplerShift`, default 100 Hz). TDL models are tapped delay lines; CDL models are clusters of rays with arrival/departure angles, seen through the antenna arrays. A, B, C have no line of sight; D, E have one. Average channel power is 1. Returns the received samples, true `H`, and perfect timing. |
| `chan.interference` | Co-channel interferer | Another uplink user in a neighbouring cell (cell ID 1): a full transmitter of its own (own bits, own pilot scrambling, same PA model) through its own fading channel. Scaled so the average interference power per occupied resource element is `SIRdB` below the wanted signal. Its random draws never change the wanted user's bits, fading or noise. `SIRdB = Inf` (default) means no interferer. |
| `chan.awgn` | Receiver thermal noise | White circular Gaussian noise, independent per antenna. `SNRdB` is Es/N0 per occupied resource element per antenna, averaged over fading: N0 = 10^(−SNRdB/10). Each time sample has variance N0/Nfft, which gives N0 per subcarrier after the receiver's (unscaled) FFT. `SNRdB = Inf` means no noise. |
| `chan.propagate` | Whole channel | fading → + interference → + noise; returns the received samples and all parts. |

**Effects to keep in mind**

- With no Doppler and a delay spread within the cyclic prefix, the receiver's FFT output is
  exactly **Y = H·X** per subcarrier. Doppler makes the channel change *within* an OFDM
  symbol, which leaks energy between subcarriers (**intercarrier interference**): about 1.6 %
  of the signal at 100 Hz and 5 % at 500 Hz (TDL-C, 30 ns). Long-delay models at 300 ns can
  also exceed the CP and cause inter-symbol interference.
- **CDL antenna polarization:** CDL rays are cross-polarized. The toolbox default slants a
  single antenna element at 45°, which captures only about **half** the channel power
  (measured 0.50–0.60). `chan.fading` uses vertical elements at both ends, which restores an
  average power of 1, so `SNRdB` means the same for TDL and CDL. The earlier code used the
  default, so its CDL results ran at roughly 2.5–3 dB lower SNR than configured.

### Channel settings (`simo_config`)

| Setting | Default | Meaning |
|---|---|---|
| `ChannelModel` | `'TDL-C'` | `'TDL-A'`…`'TDL-E'`, `'CDL-A'`…`'CDL-E'` |
| `DelaySpread` | 300e-9 | RMS delay spread, s |
| `MaximumDopplerShift` | 100 | Hz; 0 is a static channel |
| `CarrierFrequency` | 3.5e9 | Hz (CDL only) |
| `NumReceiveAntennas` | 2 | Receive antennas |
| `SNRdB` | 10 | Es/N0 per resource element per antenna; `Inf` = no noise |
| `SIRdB` | `Inf` | Signal-to-interference ratio; `Inf` = no interferer |
| `Seed` | 42 | Fixes the fading realization (and the interferer) |

## Receiver (`+rx`)

```
 received   ┌──────┐   ┌──────────┐   ┌─────────┐   ┌─────────────┐   ┌──────────┐   ┌───────┐   ┌──────┐
 samples  → │ sync │ → │ CP + FFT │ → │ LS at   │ → │ interpolate │ → │ equalize │ → │ soft  │ → │ LDPC │ → bits
            └──────┘   └──────────┘   │ pilots  │   └─────────────┘   │ (MRC)    │   │ demap │   │ dec. │
                                      └─────────┘                     └──────────┘   └───────┘   └──────┘
```

The whole link for one slot:

```matlab
cfg = simo_config('Coding','ldpc','SNRdB',12);
ctx = rx.context(cfg);                          % what the receiver knows (build once)
[rxWaveform, truth] = run_link(cfg);            % Tx + channel
llr = rx.receive(rxWaveform, ctx, 'conventional');
result = evaluate(llr, truth, cfg)              % uncoded errors, coded errors, block error
```

**What the receiver may use.** `rx.context(cfg)` holds everything known before a slot arrives,
built from the configuration only: numerology, data/pilot positions, pilot values, the FFT
window, and fixed interpolation matrices. The conventional receiver uses only `rxWaveform` and
`ctx`. `truth` (true H, timing, bits) is for evaluation and for the explicit genie baseline
`'knownH'`, which must be handed the truth: `rx.receive(rxWaveform, ctx, 'knownH', truth)`.

| Function | Physical block | What it does |
|---|---|---|
| `rx.context` | Receiver configuration | Everything the receiver knows in advance (see above). |
| `rx.sync` | Timing synchronization | Finds where the slot starts in the received samples. `'dmrs'` (default) correlates each antenna with the known pilot-only waveform and takes the strongest lag — the strongest path. `'perfect'` uses the true timing (genie). Measured: within 4 samples of the truth in all tested cases, 91 % exact, and the FFT window tolerates 18. |
| `rx.ofdm_demod` | CP removal + FFT | For each symbol, an Nfft-sample window starting halfway into the CP (so small timing errors stay inside the CP), FFT, keep the 312 occupied bins, undo the phase rotation from the early window. Output `Y`: 312 × 14 × 2. Equals `nrOFDMDemodulate`. |
| `rx.ls_estimate` | Channel estimation at the pilots + noise estimate | Least squares: Ĥ = Y · conj(X) at each pilot (unit-power pilots). Noise (plus interference) power from second differences of neighbouring pilot estimates, E\|d\|² = 6σ². |
| `rx.interpolate` | Pilots → full grid | Linear in frequency between the LS pilot estimates (extrapolated at the edges), linear in time between pilot symbols, nearest pilot symbol held outside them. With one pilot symbol the channel is assumed constant over the slot, so Doppler is not tracked. |
| `rx.equalize` | SIMO combining + equalization | LMMSE equalization: w = hᴴ/(‖h‖² + σ²), with its bias μ = ‖h‖²/(‖h‖² + σ²) removed for the demapper. For one stream this equals maximum-ratio combining, x̂ = Σ conj(H)·Y / Σ\|H\|², with noise variance ν = σ² / Σ\|H\|² — large where the channel fades. |
| `rx.demap_llr` | Soft demapper | Max-log LLR per bit, `log P(0)/P(1)`, in transmit bit order: (closest point with bit 1 − closest with bit 0)/ν. Equals `nrSymbolDemodulate`. |
| `rx.ldpc_decode` | Channel decoder | Rate recovery → belief-propagation LDPC decoding (`LDPCMaxIterations`, default 25) → reassembly → CRC check. Returns the transport block and whether its CRC passed. |
| `rx.receive` | Whole receiver | `'conventional'`: the chain above (LS at the pilots, linear interpolation, LMMSE equalization). `'knownH'`: true H instead of estimation and interpolation, with true timing (the true H belongs to the true timing: a one-sample shift rotates subcarrier k by 2πk/Nfft). Returns LLRs and every intermediate result. |

**Matches the paper's baseline.** Pihlajasalo et al. (IEEE TWC 2023, Sec. V) compare against
two LMMSE receivers: one with an LS channel estimate at the DMRS, interpolated linearly over the
slot (nearest pilot symbol held beyond the last one), and one with the known channel.
`'conventional'` and `'knownH'` are these two. "LMMSE" there names the equalizer, not the
channel estimator. One difference remains: the paper uses two DMRS symbols per slot, this
project's default is one (`DMRSAdditionalPosition` 0).

**Neural receivers can train through these blocks.** `ofdm_demod`, `ls_estimate`,
`interpolate`, `equalize` and `demap_llr` use only indexing, FFT, products and `min`, work on
batches (a 4th dimension), and run on `dlarray` with gradients (R2026b). `sync` is not
differentiable and sits before any learned block.

**Top-level helpers**

| File | What it does |
|---|---|
| `run_link.m` | Tx + channel for one slot. Separate random streams from `cfg.Seed`: bits, fading, interferer, noise — so changing the coding never changes the fading or noise. Returns the received waveform, `truth`, and all Tx/channel checkpoints. |
| `evaluate.m` | Uncoded bit errors (hard decisions on the LLRs), and with LDPC the decoded bit errors and block error (CRC). |

**Reference performance** (TDL-C, 300 ns, 100 Hz Doppler, 16QAM, LDPC rate 0.5, one DMRS
symbol, DMRS timing; 30 slots per point):

| SNR (dB) | 0 | 4 | 8 | 12 | 16 | 20 | 24 | 28 |
|---|---|---|---|---|---|---|---|---|
| Uncoded BER, known H | 0.24 | 0.15 | 0.069 | 0.021 | 0.0040 | 6e-4 | 7e-5 | 4e-6 |
| Uncoded BER, LS | 0.32 | 0.22 | 0.13 | 0.060 | 0.027 | 0.017 | 0.013 | 0.012 |
| BLER, known H | 1 | 0.43 | 0.03 | 0 | 0 | 0 | 0 | 0 |
| BLER, LS | 1 | 1 | 0.27 | 0.03 | 0 | 0 | 0.03 | 0.03 |

The LS receiver levels off near BER 0.012 because a single pilot symbol cannot follow the channel as it changes over the slot (Doppler).
This gap to known H is what the neural receivers aim to close.

### Receiver settings (`simo_config`)

| Setting | Default | Meaning |
|---|---|---|
| `TimingEstimation` | `'dmrs'` | `'dmrs'` (estimated) or `'perfect'` (genie) |
| `LDPCMaxIterations` | 25 | LDPC decoder iterations |

## Neural receivers (`+nn`)

### FCNN (DeepRx)

The FCNN replaces channel interpolation, equalization and demapping with one fully
convolutional network over the time–frequency grid. It shares the receiver front end
(`rx.sync`, `rx.ofdm_demod`) and the decoder and scoring (`rx.ldpc_decode`, `evaluate`) with
the conventional receiver, so all receivers are compared on identical received signals.

```
rx.sync → rx.ofdm_demod → nn.deeprx_features → network → nn.to_llr → evaluate / rx.ldpc_decode
```

```matlab
s = load('models/fcnn.mat');                       % from train_fcnn
llr = nn.receive(rxWaveform, rx.context(cfg), s.model);
result = evaluate(llr, truth, cfg);
```

| File | Role |
|---|---|
| `nn.deeprx_features` | Network input, F × N × 10 for 2 antennas: real then imaginary parts of [Y, pilot grid Xp, raw LS Hraw = Y·conj(Xp)] — the raw channel at the pilots, zero elsewhere, *not* interpolated. Same channel order as the earlier `deeprx_sample.m`. Runs on `dlarray`, so a TCNN in front of it can train through it. |
| `nn.labels` | Training targets: the coded bits on the grid as Q bit planes (16QAM: real sign, imaginary sign, real level, imaginary level), and a mask that excludes pilots and empty resource elements. |
| `nn.fcnn` | The paper's network (Table I): 3×3 input convolution, 11 residual blocks of depthwise-separable dilated convolutions (widths 64→256→64, dilations up to 6 in frequency and 3 in time, so each output sees far enough to reach the pilots), 3×3 output convolution. 1,234,436 parameters. Widths and dilations are optional arguments. |
| `nn.to_llr` | Network output → LLRs in transmit bit order. The network outputs the logit of P(bit = 1); **LLR = −logit**. This is the only place the convention is converted. |
| `nn.bce_loss` | Masked binary cross-entropy from logits (numerically stable; exact gradient σ(s) − t). |
| `nn.plan_slots`, `nn.slot_cfg` | Conditions per training slot, drawn uniformly from the `nn_config` ranges: SNR, Doppler, delay spread, channel model, pilot layout, SIR, PA back-off, and a PA from the bank. Every slot has a unique seed; training and validation never share one. |
| `nn.pa_bank` | Training and unseen validation sets of slightly different polynomial PAs (see *Training under PA distortion*). |
| `nn.make_data` | Simulates planned slots (`run_link` + `rx.sync`) and stores the synced time samples, pilot grids, labels and masks (and the truth for BLER on validation sets). Storing time samples lets FCNN and TCNN use the same data. |
| `nn.fcnn_forward` | FFT → features → network, the forward function the training loop calls. |
| `nn.train` | Generic training loop for every neural receiver: Adam, mini-batches, fresh training slots each epoch, validation loss / BER / BLER each epoch, keeps the best checkpoint. `params` is a struct of networks, so TCNN and hybrids reuse it; networks can be frozen. |
| `nn.receive` | The FCNN receiver; same inputs and outputs as `rx.receive`. Uses only the received samples and `ctx`. |
| `nn_config.m` | Training settings (condition ranges, seeds, epochs, batch size, learning rate, architecture), separate from the physical link in `simo_config`. |
| `train_fcnn.m` | Plan → simulate validation set → build → train → save `models/<Name>.mat` (network, link and training configuration, feature layout, output convention, best epoch). |

**Training**

```matlab
train_fcnn                                   % defaults: TDL-C, SNR -4..32 dB, Doppler 0..500 Hz,
                                             % delay 10-300 ns, SIR 0..36 dB, 150 epochs x 1000 slots
train_fcnn('MaxEpochs',20,'Name','fcnn_quick')
train_fcnn('ChannelModels',{'TDL-A','TDL-C','TDL-E'},'SIRRange',[Inf Inf])
```

Default training simulates about 1000 fresh slots per epoch; on a GPU one epoch takes on the
order of 1–2 minutes, so 150 epochs take a few hours. The network learns slowly at first
(BER stays near 0.5 for the first few epochs, as in the earlier runs).

**The earlier trained model still works.** `reference/legacy_fcnn.mat` (from the earlier
`network.m`) uses the same features and output convention. In the new pipeline, on 20 new
TDL-C slots at 20 dB and 100 Hz, it reaches uncoded BER 0.0025 against 0.0165 for the
conventional (LS) receiver. Models trained on CDL channels with the earlier code should be retrained (see the
CDL polarization note above).

### Training the FCNN yourself

1. Open MATLAB in this folder (needs 5G Toolbox and Deep Learning Toolbox; a GPU is used
   automatically through Parallel Computing Toolbox when available).
2. Optional, about 1 minute: check the pipeline first.
   ```matlab
   test_nn
   ```
3. Start training:
   ```matlab
   [model, history] = train_fcnn;                       % defaults, saves models/fcnn.mat
   [model, history] = train_fcnn('Name','fcnn_tdl_mix', ...
       'ChannelModels',{'TDL-A','TDL-C','TDL-E'});      % any nn_config setting
   ```
   Or without the desktop, logging to a file:
   ```bash
   matlab -batch "train_fcnn" > train_fcnn.log 2>&1
   ```
4. Each epoch prints training loss and validation loss, uncoded BER and LDPC BLER on the fixed
   validation slots. BER sits near 0.5 for the first few epochs before it drops.
5. `models/<Name>.mat` is rewritten whenever validation loss improves, so stopping early
   (Ctrl+C) keeps the best model so far. It holds `model` (network, link and training
   configuration, feature layout, output convention, best epoch) and `history`.
   Training cannot be resumed from a checkpoint.
6. Use the model:
   ```matlab
   s = load('models/fcnn.mat');
   cfg = simo_config('Coding','ldpc','SNRdB',20,'Seed',5001);
   ctx = rx.context(cfg);
   [rxWaveform, truth] = run_link(cfg);
   rFCNN = evaluate(nn.receive(rxWaveform, ctx, s.model), truth, cfg)
   rLS   = evaluate(rx.receive(rxWaveform, ctx, 'conventional'), truth, cfg)
   ```

Run time: the defaults simulate 1000 fresh slots per epoch for 150 epochs; with a GPU an epoch
takes roughly 1–2 minutes (slot simulation is a large part of it), about 3–5 hours in total.
For a first look, `train_fcnn('MaxEpochs',20,'Name','fcnn_quick')` takes about 30 minutes.
Main settings (`nn_config`): `MaxEpochs` (150), `SlotsPerSeed` (200 per training seed per
epoch, 5 seeds), `MiniBatchSize` (4), `LearningRate` (1e-3), `SNRRange` ([-4 32] dB),
`DopplerRange` ([0 500] Hz), `DelaySpreadRange` ([10 300] ns), `SIRRange` ([0 36] dB;
`[Inf Inf]` for no interferer), `ChannelModels` ({'TDL-C'}), `PilotConfigs`, `Link` (simo_config
overrides for the base link, default `{'Coding','ldpc'}`), `Widths`/`Dilations`, `Name`.

### Training under PA distortion

Real PAs of one type differ slightly, so a receiver should learn to undo a *family* of PAs,
not one curve. With a polynomial PA in the `Link`, `nn.pa_bank` draws 30 training PAs and 10
validation PAs. Each one is the Link's coefficients `[1 -1/3]` plus complex Gaussian dither of
10% of each coefficient, and the dither also adds AM/PM. A draw is kept only if its AM/AM
curve rises up to saturation. Each slot draws one PA from its bank and a back-off, so
validation always measures **unseen** PAs. The plan records each slot's `PAIndex` and
`PACoefficients`, and the model saves the bank (`model.PABank`).

| Back-off | 0 dB | 3 dB | 6 dB | 10 dB |
|---|---|---|---|---|
| EVM (average over PAs) | 32% | 22% | 12% | 4.6% |

`experiments/pa_common.m` holds the shared PA settings: polynomial PA, back-off 0–10 dB, the
PA bank and no interferer. Every PA experiment (FCNN now, TCNN and hybrids later) uses them,
so the receivers are compared under identical conditions. Change a setting there and every
run follows.

```matlab
addpath experiments
args = pa_common;                               % {:} needs a variable, not a function call
train_fcnn(args{:}, 'Name', 'fcnn_pa')          % FCNN under PA distortion, ~3 h
```

Bank settings in `nn_config`: `PADither` (0.1; 0 uses the Link's PA for every slot),
`NumTrainingPAs` (30), `NumValidationPAs` (10), `PASeed` (2023). Without a polynomial PA,
plans and training are exactly as before.

### TCNN and hybrid flows

A **TCNN** is a time-domain CNN. It cleans the received **time samples** before CP removal and
the FFT, which is where a PA's sample-by-sample distortion is easiest to undo, before the FFT
spreads it over all subcarriers. What follows it is an ordinary receiver. A *flow* names the
whole chain (`nn.flow`):

```
rx.sync ─► slot ─► TCNN: slot + correction ─► cleaned slot
                                                 │
        ┌────────────────────────────────────────┴───────────────────────────┐
  'tcnn_feq'                                                          'tcnn_fcnn'
  rx.ofdm_demod → ls_estimate → interpolate → equalize → demap_llr    rx.ofdm_demod → deeprx_features → FCNN
        └──────────────────────────────► LLRs ◄──────────────────────────────┘
'fcnn' (DeepRx): rx.sync → rx.ofdm_demod → deeprx_features → FCNN → LLRs
```

The TCNN's last layer starts at **zero**, so an untrained TCNN passes the samples through
unchanged. An untrained `tcnn_feq` *is* the conventional receiver, and an untrained `tcnn_fcnn` *is* the
FCNN it starts from. In `tcnn_feq` only the TCNN learns. The loss is the same bit
cross-entropy, on the conventional receiver's LLRs (logit = −LLR), and the gradients pass back
through the fixed FFT, LS estimation, linear interpolation, LMMSE equalizer and demapper.

| File | Role |
|---|---|
| `nn.tcnn` | The network: residual blocks of depthwise-separable 3-tap convolutions along time (widths 64-128-256-128-64, dilations 1-2-4-2-1, so each output sees 42 samples ≈ 5.5 µs, enough for the channel's delay spread). Its output is input + correction, with the correction layer starting at zero. |
| `nn.tcnn_apply` | Slot (L × Nr × B complex) → network input L × 1 × [Re Im per antenna] → cleaned slot. Each slot is scaled to unit RMS for the network and scaled back afterwards. |
| `nn.flow`, `nn.flow_tcnn_feq`, `nn.flow_tcnn_fcnn` | The flows: which networks each uses and its forward pass. A new flow is one case in `nn.flow` plus one forward function. |
| `nn.to_planes` | LLRs in transmit order → bit planes on the grid, so the conventional demapper's output uses the FCNN's loss and labels. |
| `nn.in_training` | Frozen networks run in prediction mode, so their batch-norm statistics stay fixed. |
| `train_receiver.m` | `train_fcnn` for any flow. `'Flow'` picks the chain, `'InitFrom'` loads networks from a saved model, and `'Freeze'` keeps named networks fixed. It saves `model.Flow`, `model.Nets` (one field per network) and `model.Trainable`. |
| `nn.receive` | Runs any saved model's flow (models without `Flow` are FCNNs). |

**Training the PA experiments** (shared settings from `experiments/pa_common.m`):

```matlab
addpath experiments
args = pa_common;
train_fcnn(args{:}, 'Name', 'fcnn_pa')                               % FCNN alone
train_receiver(args{:}, 'Flow', 'tcnn_feq', 'Name', 'tcnn_feq')      % TCNN + conventional
train_receiver(args{:}, 'Flow', 'tcnn_fcnn', 'Name', 'tcnn_fcnn', ...
    'InitFrom', 'models/fcnn_pa.mat', 'Freeze', {'fcnn'})            % TCNN in front of the trained FCNN
```

To fine-tune both networks of the hybrid afterwards, run `train_receiver(args{:},
'Flow','tcnn_fcnn', 'Name','tcnn_fcnn_ft', 'InitFrom','models/tcnn_fcnn.mat',
'LearningRate',1e-4)`. Without `Freeze`, both networks learn.

The TCNN works on 7680 samples × 2 antennas per slot. Expect an epoch to take about as long as
the FCNN's, because slot simulation dominates.

#### Relation to the paper, and two TCNN architectures: `'TCNNArchitecture'`

The flows follow Pihlajasalo et al., "Deep Learning OFDM Receivers for Improved Power
Efficiency and Coverage", *IEEE Trans. Wireless Commun.* 22(8), 2023:

| Here | Paper |
|---|---|
| `'fcnn'` | DeepRx (FCNN), Sec. III-B, the benchmark |
| `'tcnn_feq'` | **TCNN/FEQ**, Sec. III-C: time-domain CNN + conventional frequency-domain receiver, trained end to end through the untrainable FFT and receiver blocks |
| `'tcnn_fcnn'` | **HybridDeepRx (TCNN/FCNN)**, Sec. III-D: the same TCNN and the DeepRx CNN, joined by the FFT, trained jointly |

The conventional receiver (the FEQ) is the paper's: LS at the DMRS, linear interpolation,
LMMSE equalization. The paper uses two DMRS symbols per slot, this project one. The
paper's purely frequency-domain variant (FTF-CNN, Sec. III-E) is not implemented.

| | `'residual'` (default, `nn.tcnn`) | `'paper'` (`nn.tcnn_paper`, Sec. III-C) |
|---|---|---|
| Input | the whole slot: 7680 × 1 × [Re Im of both antennas] | per antenna: 552 × 14 × [Re Im] (Z_pre); one column per OFDM symbol including its CP, shorter CPs zero-padded in front; the antennas share the network |
| Convolutions | depthwise-separable 3-tap along time, depth multiplier 2, dilations 1-2-4-2-1 | depthwise-separable 3×3 over (sample, symbol), depth multiplier 1, so neighbouring symbols are seen too |
| Blocks | 5 pre-activation blocks, 64-128-256-128-64, batch norm | the same: 5 pre-activation blocks, 64-128-256-128-64, batch norm |
| Output | input + correction; the correction starts at zero (untrained = pass-through) | 1×1 conv to 2 channels, no activation: the output *is* the cleaned signal |
| Parameters | ≈ 0.48 M | ≈ 0.29 M (paper: "about 280 thousand" for TCNN/FEQ) |

The paper gives the layout, the five blocks and their widths, the 3×3 filters and the last
1×1 layer. The rest of `'paper'` follows DeepRx, whose pre-activation blocks the paper says it
uses: per block, twice BN → ReLU → depthwise 3×3 → pointwise 1×1; 1×1 skip projections when the
width changes; a 3×3 input convolution; BN → ReLU before the last layer; and the per-slot RMS
scaling. Separable convolutions are an inference from the paper's parameter count: plain 3×3
convolutions at these widths would need about 2.4 M parameters. `TCNNWidths` changes the blocks
(no `TCNNDilations` for `'paper'`).

```matlab
% TCNN/FEQ with the paper's TCNN
train_receiver(args{:}, 'Flow','tcnn_feq', 'TCNNArchitecture','paper', 'Name','tcnn_feq_paper')
% HybridDeepRx: the paper's TCNN + FCNN, trained together from scratch
train_receiver(args{:}, 'Flow','tcnn_fcnn', 'TCNNArchitecture','paper', 'Name','hybrid_paper')
```

An untrained paper TCNN does **not** pass the samples through, so `tcnn_feq_paper` starts
worse than the conventional receiver and has to learn the identity first. Models saved before this option count as
`'residual'`.

The paper also trains differently: TDL-B/C/D for training and TDL-A/E for validation,
back-off −1 to 7 dB, a 17th-order polynomial fitted to a measured PA, batch 20, and learning
rate 1e-3 with warm-up and linear decay. `pa_common` keeps this project's link: [1 −1/3] PA,
back-off 0–10 dB, TDL-C with randomized Doppler and delay spread, batch 4, constant learning rate. The PA bank (30
training and 10 validation PAs, dither proportional to each coefficient, rising and concave
check) follows the paper.

## Training curves: `plot_history`

Every model file stores its per-epoch `history`: training loss, validation loss, BER, BLER and
seconds. `plot_history` plots it for any set of models:

```matlab
plot_history({'fcnn','fcnn_pa','tcnn_feq'})
plot_history({'fcnn','fcnn_quick'}, 'Labels',{'150 epochs','20 epochs'}, 'Name','fcnn_runs')
```

It makes three panels, one colour per model:
1. **Loss:** training dashed, validation solid. If both are still falling, more epochs would
   help. If the validation loss rises while the training loss falls, the model is
   overfitting.
2. **Validation BER** before LDPC, on a log scale.
3. **Validation BLER** after LDPC.

The circle marks the best epoch, which is the network that was saved. It prints a summary
table (epochs, best epoch, losses, BER and BLER at the best epoch, hours) and saves
`results/<Name>.png` (default `training_history`). A run still in progress saves its history
only up to its latest best epoch.

Validation sets differ between experiments, e.g. `fcnn` has interferers and `fcnn_pa` has
PAs, so compare their curves for **shape**, not level. Use `compare_snr` to compare receivers
on the same slots.

## Comparing receivers: `compare_snr`

`compare_snr` runs any trained models, from `train_fcnn` or `train_receiver` (any flow) or the
legacy file, side by side with the paper's two LMMSE baselines: LS-estimated channel ('LMMSE') and known channel ('LMMSE known H'). All receivers see **the same
slots**, and every point gives three numbers: uncoded BER (hard decisions before LDPC, the
receiver itself), coded BER (information bits after LDPC decoding) and BLER (blocks whose CRC
fails, what a real link retransmits). With only 30 slots, coded BER usually drops from about
0.1 to exactly 0 within a few dB, because LDPC either decodes a block fully or fails badly.

```matlab
compare_snr({'fcnn','fcnn_pa','tcnn_feq'})                          % SNR 0:4:32 dB, 30 slots per point
compare_snr({'fcnn_pa','tcnn_feq'}, 'Link',{'PABackoffdB',0}, 'Name','snr_pa0')   % heavy PA
compare_snr({'fcnn','reference/legacy_fcnn.mat'}, 'Labels',{'new','legacy'}, ...
    'SNRs',-4:2:32, 'Slots',100, 'Name','snr_new_vs_legacy')
```

- **Models:** a name means `models/<name>.mat`. Anything ending in `.mat` is a file path.
- **Test link:** the **first** model's link (PA, Doppler, delay spread, channel model, SIR, …),
  changed with `'Link',{...}` (simo_config names). Put the model whose conditions you want
  first, or set them explicitly.
- **PA:** with a polynomial PA, slots cycle through the 10 **unseen validation PAs** saved in
  the models, so no model is tested on a PA it trained on. The back-off is the link's
  `PABackoffdB`, default 6 dB.
- **Layout check:** a model trained for a different modulation, number of RBs, antennas or
  DMRS layout is rejected with a message naming the setting.
- **Output:** a line per SNR point while it runs, then `results/<Name>.png` (uncoded BER, coded
  BER and BLER vs SNR; a missing point means zero errors) and `results/<Name>.mat` (the results table, the
  conditions and the options).

Options: `'SNRs'` (0:4:32), `'Slots'` (30), `'Seed'` (5001, the first slot seed), `'Link'`,
`'UsePABank'` (true), `'Labels'`, `'Name'` ('snr_sweep'). It runs on the CPU, so it can run
next to a GPU training. With 30 slots, BLER moves in steps of 0.033. Use more slots for smooth
BLER curves.

## Tests

```matlab
test_tx          % transmitter checks
test_chan        % channel checks
test_rx          % receiver checks, end to end with LDPC
test_nn          % neural-receiver pipeline checks (about 1 min with a GPU)
test_tcnn        % TCNN flows (CPU only, so it can run beside a GPU training)
```

`test_chan` checks the channel against the earlier code (identical timing, `H`, noiseless and
interference signals for the TDL cases; noise power only, since noise now has its own random
stream; CDL is skipped because of the polarization fix) and against physics: Y = H·X for a
static channel and growing intercarrier interference with Doppler, average channel power of 1
for TDL and CDL models, measured SIR and SNR equal to the settings, and an unchanged wanted
signal when interference is added.

`test_tx` checks three things:

1. **Uncoded Tx matches the earlier code:** with `Coding` `'none'`, `tx.transmit`
   reproduces the bits, grid, masks and PA gain saved from the earlier transmitter for
   six slot cases.
2. **The LDPC encoder matches MATLAB's `nrULSCH` bit for bit**, for QPSK and 16QAM, code
   rates 0.3–0.8, and a 106-RB slot with 7 code blocks.
3. **Coded loopback over noise:** Tx → OFDM demodulation → AWGN → soft demapping → LDPC
   decoding. Without noise the transport block is recovered exactly. With noise, BLER drops
   from 1 to 0 between 4 and 6 dB SNR (16QAM, rate 0.48; the theoretical limit is
   ≈ 4.4 dB).

`test_rx` checks the receiver against the earlier `decode_simo_ls` (channel estimate, noise
power, equalized symbols and hard decisions identical to ~1e-16 for six saved slots),
`rx.ofdm_demod` against `nrOFDMDemodulate` and `rx.demap_llr` against `nrSymbolDemodulate`,
DMRS timing against the true timing (and the BER cost of the difference) for five channel
models at 0/10/30 dB, `dlarray` results, gradients and batches, and the end-to-end SNR sweep
above (BLER from 1 to ~0, known H never worse than LS).

`test_nn` checks the neural pipeline before any long training: an oracle round trip
(labels → perfect logits → LLRs → LDPC decoding gives zero errors, so bit order and sign are
right), features and labels identical to the earlier `deeprx_sample.m`, the loss gradient, the
paper's network (same parameter count as the earlier one) with gradients reaching the received
samples, a short training run on 8 slots in which loss and BER fall steadily, and the earlier
trained network beating the conventional receiver in the new pipeline.

`test_tcnn` checks the TCNN flows before any long training:
- an untrained `tcnn_feq` gives the conventional receiver's LLRs;
- the bit loss reaches every TCNN weight through the fixed conventional receiver;
- a small TCNN learns to undo heavy PA distortion (0 dB back-off) on 8 slots, beating the conventional receiver on
  loss and BER;
- `nn.receive` reproduces the training forward pass;
- an untrained TCNN in front of an FCNN gives that FCNN's LLRs;
- `train_receiver` with `InitFrom` and `Freeze` leaves the frozen FCNN unchanged;
- the paper TCNN's per-symbol layout is exact (an identity network returns the slot),
  gradients reach all its weights, and `train_receiver` / `nn.receive` run it.

`test_tx`, `test_chan`, `test_rx` and `test_nn` compare against `reference/ref_v0.mat`
(saved from the earlier code) where the new code is meant to agree with it.
