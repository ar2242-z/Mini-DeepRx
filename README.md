# Mini DeepRx: link-level simulation in MATLAB

A 5G uplink link-level simulation, currently one transmit antenna and two receive
antennas (SIMO, single stream; `NumReceiveAntennas` sets the receive side), built to
compare a conventional receiver with neural receivers
(TCNN, FCNN/DeepRx, and hybrids) on identical signals.

The code is organised along the signal chain. Each physical processing block is
one function, grouped by where it sits in the chain:

```
+tx/    transmitter        bits → LDPC → symbol mapping → resource grid → IFFT + CP → PA
+chan/  channel            fading → interference → noise
+rx/    receiver           sync → FFT → channel estimation → equalization → LLRs → LDPC decoding
+nn/    neural receivers   FCNN (DeepRx), TCNN, hybrids (TCNN/FCNN, FTF-CNN) and training
+sim/   simulation glue    run_link (Tx + channel for one slot), evaluate (BER/BLER), override (config)
```

All blocks share one configuration struct from `link_config.m`. The earlier scripts
(`mini_simo_channel.m`, `decode_simo_ls.m`, `network.m`, `TCNN.m`, `sweep.m`, …) were removed;
they remain in git history (commit `b29a596`), as do the reference outputs the rebuild was
checked against.

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
cfg = link_config('Coding','ldpc');                 % or 'none' for uncoded bits
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
| `tx.miniPA` | RF power amplifier (miniPA) | Memoryless non-linear amplifier (Rapp, or a polynomial with optional AM/PM). OFDM has a high peak-to-average power ratio, so peaks are compressed; **back-off** (`PABackoffdB`) sets the average level below saturation. The output is rescaled to the input power, so only distortion is added. `PAGain` is the Bussgang gain (output = gain × input + uncorrelated distortion). `PAModel` `'none'` is a linear PA. |
| `tx.mppa` | RF power amplifier with memory (`PAModel` `'mp3gpp'`) | The 3GPP TR 38.803 memory polynomial PA (31 GHz CMOS, K = 8, M = 5), run at 4× the sample rate, with **output back-off relative to its 1 dB compression point**. `PAGain` is then one Bussgang gain per subcarrier (F × 1). See [MPPA.md](MPPA.md). |
| `tx.transmit` | Whole transmitter | Runs the blocks above in order and returns the waveform and every intermediate result. |
| `tx.bits_per_symbol` | Helper | QPSK → 2, 16QAM → 4, 64QAM → 6. |

Not modelled: data scrambling (TS 38.211 6.3.1.1), transmit windowing, and
up-conversion to the carrier frequency (all signals are complex baseband).

### Transmitter settings (`link_config`)

| Setting | Default | Meaning |
|---|---|---|
| `NumResourceBlocks` | 26 | Bandwidth, 12 subcarriers per block |
| `SubcarrierSpacing` | 15 | kHz |
| `Nfft` | 512 | FFT size; must exceed 12 × `NumResourceBlocks` |
| `Modulation` | `'16QAM'` | `'QPSK'`, `'16QAM'` or `'64QAM'` |
| `Coding` | `'none'` | `'ldpc'` for 5G UL-SCH coding |
| `TargetCodeRate` | 0.5 | Information bits / coded bits (LDPC only) |
| `DMRSTypeAPosition`, `DMRSLength`, `DMRSAdditionalPosition`, `DMRSConfigurationType`, `NumCDMGroupsWithoutData` | 2, 1, 0, 1, 1 | DMRS pilot pattern (TS 38.211 6.4.1.1); default is one pilot symbol at MATLAB column 3 on every 2nd subcarrier |
| `PAModel` | `'none'` | `'none'`, `'rapp'` or `'polynomial'` (miniPA), `'mp3gpp'` (memory polynomial, [MPPA.md](MPPA.md)) |
| `PABackoffdB` | 6 | miniPA: input RMS below PA saturation, dB. `'mp3gpp'`: mean output power below the output power at P1dB, dB |
| `RappSmoothness`, `PACoefficients` | 2, `[1 -1/3]` | miniPA parameters |
| `MPCoefficients`, `PAOversampling` | `[]`, 4 | `'mp3gpp'` parameters: normalized K × M coefficients as a row (`[]` = nominal 3GPP PA), oversampling of the PA |
| `Seed` | 42 | Random seed for the slot |

```matlab
cfg = link_config('Modulation','QPSK','Coding','ldpc','TargetCodeRate',0.3, ...
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

### Channel settings (`link_config`)

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
cfg = link_config('Coding','ldpc','SNRdB',12);
ctx = rx.context(cfg);                          % what the receiver knows (build once)
[rxWaveform, truth] = sim.run_link(cfg);            % Tx + channel
llr = rx.receive(rxWaveform, ctx, 'conventional');
result = sim.evaluate(llr, truth, cfg)              % uncoded errors, coded errors, block error
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
| `rx.equalize` | Receive combining + equalization (single stream) | LMMSE equalization: w = hᴴ/(‖h‖² + σ²), with its bias μ = ‖h‖²/(‖h‖² + σ²) removed for the demapper. For one stream this equals maximum-ratio combining, x̂ = Σ conj(H)·Y / Σ\|H\|², with noise variance ν = σ² / Σ\|H\|² — large where the channel fades. |
| `rx.demap_llr` | Soft demapper | Max-log LLR per bit, `log P(0)/P(1)`, in transmit bit order: (closest point with bit 1 − closest with bit 0)/ν. Equals `nrSymbolDemodulate`. |
| `rx.ldpc_decode` | Channel decoder | Rate recovery → belief-propagation LDPC decoding (`LDPCMaxIterations`, default 25) → reassembly → CRC check. Returns the transport block and whether its CRC passed. |
| `rx.receive` | Whole receiver | `'conventional'`: the chain above (LS at the pilots, linear interpolation, LMMSE equalization). `'knownH'`: true H instead of estimation and interpolation, with true timing (the true H belongs to the true timing: a one-sample shift rotates subcarrier k by 2πk/Nfft). `'dpoi'` and its variants: the PA compensation baseline, see *PA compensation baseline* below. Returns LLRs and every intermediate result. |
| `rx.delay_estimate` | Smoothed channel estimate | Fits the LS pilot estimates with delay taps inside a 2.4 µs window (ridge least squares) and evaluates them on every subcarrier. Used by the DPoI modes. |
| `rx.dpoi` | Digital Post-Inverse | Equalize → oversampled IFFT → memory polynomial fitted to the slot's pilots → FFT. See below. |

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

**Simulation helpers (`+sim`)**

| File | What it does |
|---|---|
| `sim.run_link` | Tx + channel for one slot. Separate random streams from `cfg.Seed`: bits, fading, interferer, noise — so changing the coding never changes the fading or noise. Returns the received waveform, `truth`, and all Tx/channel checkpoints. |
| `sim.evaluate` | Uncoded bit errors (hard decisions on the LLRs), and with LDPC the decoded bit errors and block error (CRC). |
| `sim.override` | Copies settings onto a configuration and re-validates it with `link_config` (per-slot pilot layouts, test links from a saved model). |

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

### Receiver settings (`link_config`)

| Setting | Default | Meaning |
|---|---|---|
| `TimingEstimation` | `'dmrs'` | `'dmrs'` (estimated) or `'perfect'` (genie) |
| `LDPCMaxIterations` | 25 | LDPC decoder iterations |

### PA compensation baseline: Digital Post-Inverse (`'dpoi'`)

The LMMSE receivers above do nothing about the PA, so a neural receiver's gain over them mixes
PA compensation with better channel estimation. `'dpoi'` is a conventional receiver that does
compensate the PA: the Digital Post-Inverse of Babaroglu et al., "Cellular Digital
Post-Distortion: Signal Processing Methods and RF Measurements", IEEE WAMICON 2023. It is the
model-based counterpart of the FTF-CNN: equalize, go to time, undo the PA, come back.

```
rx.sync → rx.ofdm_demod → rx.ls_estimate → rx.delay_estimate → LMMSE equalization (whole grid)
        → 4×Nfft IFFT per symbol → memory polynomial x(n) = Σ c(p,d) q(n−d)|q(n−d)|^(p−1)
        → FFT → rx.demap_llr
```

The coefficients c (orders 1, 3, 5, 7 and delays 0–4: 20 complex numbers) are fitted by least
squares once per slot so that the output equals the known pilots. The receiver uses only the
received samples and `ctx`: no PA model, no training, no decisions.

```matlab
llr = rx.receive(rxWaveform, ctx, 'dpoi');
[llr, trace] = rx.receive(rxWaveform, ctx, 'dpoi', [], struct('Memory', 0));   % rx.dpoi options
```

| Mode | Channel | Coefficients fitted on | Use |
|---|---|---|---|
| `'dpoi'` | LS + delay-domain smoothing | this slot's pilots | the baseline |
| `'smoothed'` | the same | none (no post-inverse) | `'dpoi'` minus `'smoothed'` is the post-inverse alone |
| `'dpoi_knownH'` | true g·H, true timing | this slot's pilots | PA compensation without channel estimation error |
| `'dpoi_genie'` | LS + delay-domain smoothing | the true grid, every RE | the best this post-inverse can do |

**What differs from the paper**

- **The channel estimate must be smoothed.** With per-pilot LS (the conventional receiver) the
  equalized pilots equal the transmitted pilots exactly, because the PA distortion on each
  pilot is absorbed into its channel estimate, and the fit finds nothing. The paper uses an
  LMMSE channel estimate; here `rx.delay_estimate` fits the pilots with delay taps, which also
  lowers the channel error by 3–7 dB at 20 dB SNR.
- **The DMRS symbol also carries data.** The paper has a DMRS-only symbol and boosts it to
  match the data symbols' envelope. Here every second subcarrier of the DMRS symbol is data.
  The basis functions are built from the received symbol, so only the target needs known
  values: the fit is made on the pilot subcarriers in frequency (156 equations in the default
  link, 396 in the `mppa_common` link). No boosting is needed.
- **LMMSE equalizer in front, bias removed after.** The post-inverse spreads the noise of
  faded subcarriers over the whole symbol, so the equalizer holds it back, and the
  per-subcarrier gain is divided out again after the FFT. The output is also rescaled so the
  pilots come out with gain 1 (a least-squares inverse fitted on noisy input shrinks its
  output, which the paper notes).
- **Noise variance for the LLRs** is the error left at the pilots after the correction.

**Measured** (uncoded BER, static TDL-C 300 ns, 6 slots, from `test_dpoi`):

| Link | LMMSE | LMMSE smoothed | DPoI |
|---|---|---|---|
| Polynomial PA, 3 dB back-off, 16QAM, 30 dB | 0.0240 | 0.0091 | 0.0006 |
| 3GPP MP PA, 3 dB back-off, 64QAM, 30 dB | 0.0753 | 0.0599 | 0.0326 |
| No PA, 16QAM, 15 dB | 0.0166 | 0.0081 | 0.0084 |

- **The gain is at medium and high SNR.** Without noise and with the true channel, the EVM
  falls from 18.9 % to 10.1 % (polynomial PA) and from 15.0 % to 9.8 % (3GPP PA) at 3 dB
  back-off; the genie fit reaches 9.8 % and 9.2 %. The rest cannot be inverted: samples clipped
  at saturation, and the distortion outside the occupied band, which the receiver never sees.
- **At low SNR it can cost a little.** In the default link at 5 dB the uncoded BER is about
  0.005–0.01 higher than `'smoothed'`: 20 coefficients are fitted on 156 noisy pilots. The
  larger `mppa_common` link shows no loss at 10 dB.
- **Doppler limits it as it limits the LMMSE receiver.** With one DMRS symbol the channel
  estimate ages over the slot; at 100 Hz and 15 kHz spacing that error is larger than the PA
  distortion, and the post-inverse cannot help. `'dpoi_knownH'` shows the PA part alone.

**Against the FTF-CNN** (`compare_snr({'ftf_mppa'}, ...)`, `mppa_common` link, TDL-C 300 ns,
100 Hz, 10 unseen PAs, 30 slots per point; uncoded BER at 20 dB SNR):

| Output back-off | LMMSE | LMMSE smoothed | DPoI | DPoI known H | `ftf_mppa` | LMMSE known H |
|---|---|---|---|---|---|---|
| 6 dB (`results/dpoi_vs_ftf_mppa`) | 0.0510 | 0.0394 | 0.0340 | – | 0.0294 | 0.0244 |
| 2 dB (`results/dpoi_vs_ftf_mppa_bo2`) | 0.1460 | 0.1263 | 0.1036 | 0.0669 | 0.0629 | 0.1038 |

At 6 dB the PA is mild: most of the FTF's margin over LMMSE is channel estimation, and DPoI
comes within 0.005. At 2 dB the FTF is well ahead of DPoI with an estimated channel and level
with DPoI given the true channel. BLER on 30 slots moves in steps of 0.033 and the default
coded grid stops at 12 dB, below the 2 dB waterfall, so use more slots and `'CodedSNRs'` up to
about 20 dB before quoting coded gains.

`rx.dpoi` options (5th argument of `rx.receive`): `Orders` ([1 3 5 7]), `Memory` (4),
`Oversampling` (4), `Ridge` (1e-2, pull towards no correction); `DelayWindow`
([-0.4e-6 2e-6] s) and `DelayRidge` (1e-3) for `rx.delay_estimate`.

## Neural receivers (`+nn`)

### FCNN (DeepRx)

The FCNN replaces channel interpolation, equalization and demapping with one fully
convolutional network over the time–frequency grid. It shares the receiver front end
(`rx.sync`, `rx.ofdm_demod`) and the decoder and scoring (`rx.ldpc_decode`, `sim.evaluate`) with
the conventional receiver, so all receivers are compared on identical received signals.

```
rx.sync → rx.ofdm_demod → nn.deeprx_features → network → nn.to_llr → sim.evaluate / rx.ldpc_decode
```

```matlab
s = load('models/fcnn.mat');                       % from train_receiver('Flow','fcnn')
llr = nn.receive(rxWaveform, rx.context(cfg), s.model);
result = sim.evaluate(llr, truth, cfg);
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
| `nn.make_data` | Simulates planned slots (`sim.run_link` + `rx.sync`) and stores the synced time samples, pilot grids, labels and masks (and the truth for BLER on validation sets). Storing time samples lets FCNN and TCNN use the same data. |
| `nn.fcnn_forward` | FFT → features → network, the forward function the training loop calls. |
| `nn.train` | Generic training loop for every neural receiver: Adam, mini-batches, fresh training slots each epoch, validation loss / BER / BLER each epoch, keeps the best checkpoint. `params` is a struct of networks, so TCNN and hybrids reuse it; networks can be frozen. |
| `nn.receive` | Runs a saved model's flow; same inputs and outputs as `rx.receive`. Uses only the received samples and `ctx`. |
| `nn_config.m` | Training settings (condition ranges, seeds, epochs, batch size, learning rate, architecture), separate from the physical link in `link_config`. |
| `train_receiver.m` | With `'Flow','fcnn'`: plan → simulate validation set → build → train → save `models/<Name>.mat` (flow, networks, link and training configuration, output convention, best epoch). See *TCNN and hybrid flows* for its other options. |

**Training**

```matlab
train_receiver('Flow','fcnn')                % defaults: TDL-C, SNR -4..32 dB, Doppler 0..500 Hz,
                                             % delay 10-300 ns, SIR 0..36 dB, 150 epochs x 1000 slots
train_receiver('Flow','fcnn','MaxEpochs',20,'Name','fcnn_quick')
train_receiver('Flow','fcnn','ChannelModels',{'TDL-A','TDL-C','TDL-E'},'SIRRange',[Inf Inf])
```

Default training simulates about 1000 fresh slots per epoch; on a GPU one epoch takes on the
order of 1–2 minutes, so 150 epochs take a few hours. The network learns slowly at first
(BER stays near 0.5 for the first few epochs, as in the earlier runs).

Models trained on CDL channels with the earlier code should be retrained (see the CDL
polarization note above).

### Training the FCNN yourself

1. Open MATLAB in this folder (needs 5G Toolbox and Deep Learning Toolbox; a GPU is used
   automatically through Parallel Computing Toolbox when available).
2. Optional, about 1 minute: check the pipeline first.
   ```matlab
   test_nn
   ```
3. Start training:
   ```matlab
   [model, history] = train_receiver('Flow','fcnn');     % defaults, saves models/fcnn.mat
   [model, history] = train_receiver('Flow','fcnn','Name','fcnn_tdl_mix', ...
       'ChannelModels',{'TDL-A','TDL-C','TDL-E'});      % any nn_config setting
   ```
   Or without the desktop, logging to a file:
   ```bash
   matlab -batch "train_receiver('Flow','fcnn')" > train_fcnn.log 2>&1
   ```
4. Each epoch prints training loss and validation loss, uncoded BER and LDPC BLER on the fixed
   validation slots. BER sits near 0.5 for the first few epochs before it drops.
5. `models/<Name>.mat` is rewritten whenever validation loss improves, so stopping early
   (Ctrl+C) keeps the best model so far. It holds `model` (flow, networks, link and training
   configuration, output convention, best epoch) and `history`.
   `'Resume',true` continues an interrupted run from `models/<Name>_last.mat`.
6. Use the model:
   ```matlab
   s = load('models/fcnn.mat');
   cfg = link_config('Coding','ldpc','SNRdB',20,'Seed',5001);
   ctx = rx.context(cfg);
   [rxWaveform, truth] = sim.run_link(cfg);
   rFCNN = sim.evaluate(nn.receive(rxWaveform, ctx, s.model), truth, cfg)
   rLS   = sim.evaluate(rx.receive(rxWaveform, ctx, 'conventional'), truth, cfg)
   ```

Run time: the defaults simulate 1000 fresh slots per epoch for 150 epochs; with a GPU an epoch
takes roughly 1–2 minutes (slot simulation is a large part of it), about 3–5 hours in total.
For a first look, `train_receiver('Flow','fcnn','MaxEpochs',20,'Name','fcnn_quick')` takes about 30 minutes.
Main settings (`nn_config`): `MaxEpochs` (150), `SlotsPerSeed` (200 per training seed per
epoch, 5 seeds), `MiniBatchSize` (4), `LearningRate` (1e-3), `SNRRange` ([-4 32] dB),
`DopplerRange` ([0 500] Hz), `DelaySpreadRange` ([10 300] ns), `SIRRange` ([0 36] dB;
`[Inf Inf]` for no interferer), `ChannelModels` ({'TDL-C'}), `PilotConfigs`, `Link` (link_config
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
train_receiver(args{:}, 'Flow', 'fcnn', 'Name', 'fcnn_pa')   % FCNN under PA distortion, ~3 h
```

Bank settings in `nn_config`: `PADither` (0.1; 0 uses the Link's PA for every slot),
`NumTrainingPAs` (30), `NumValidationPAs` (10), `PASeed` (2023). Without a polynomial PA,
plans and training are exactly as before.

The 3GPP memory polynomial PA (`'mp3gpp'`) has its own bank and shared settings,
`experiments/mppa_common.m`: the paper's numerology (60 kHz, 1024-point FFT, 792 subcarriers,
64QAM), output back-off −1 to 7 dB relative to P1dB, dither 0.05 of the static curve. See
[MPPA.md](MPPA.md); `plot_mppa` draws the paper's Fig. 4 for it.

```matlab
addpath experiments
args = mppa_common;
train_receiver(args{:}, 'Flow', 'ftf', 'Name', 'ftf_mppa')
```

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

**LLR calibration in `tcnn_feq`.** The conventional LLRs are badly overconfident: the
demapper's noise variance ν counts only thermal noise, not the channel estimation error of one
DMRS symbol held over a slot with Doppler, nor the PA distortion. On mixed training conditions
their cross-entropy is far above ln 2 (worse than guessing). Trained through them, a TCNN
lowers the loss fastest by making the signal look noisier, so the LLRs shrink, and the BER
rises. That is what the earlier 150-epoch TCNN run did (loss 1.91 → 0.52, BER 0.177 → 0.203).
`tcnn_feq` therefore learns two extra numbers with the TCNN (`nn.llr_calibration`,
`nn.calibrate_llr`):

    LLR' = a · LLR · ν / (ν + ε)        (the demapper with noise variance (ν + ε)/a)

ε is an error floor relative to the symbol power and caps |LLR| at high SNR; a is an overall
scale. Both are positive, so the calibration never flips a bit: it only fixes the confidence,
and the TCNN's gradient has to come from getting more bits right. `test_tcnn` part 7 shows the
difference on 32 mixed-condition slots: without calibration the loss falls (0.88 → 0.64) while
the BER rises (0.1885 → 0.1963); with it the loss falls further (0.40) and the BER stays at the
conventional receiver's (0.1898; a = 0.27, ε = 0.022). Models saved before the calibration
existed run without it.

| File | Role |
|---|---|
| `nn.tcnn` | The network: residual blocks of depthwise-separable 3-tap convolutions along time (widths 64-128-256-128-64, dilations 1-2-4-2-1, so each output sees 42 samples ≈ 5.5 µs, enough for the channel's delay spread). Its output is input + correction, with the correction layer starting at zero. |
| `nn.tcnn_apply` | Slot (L × Nr × B complex) → network input L × 1 × [Re Im per antenna] → cleaned slot. Each slot is scaled to unit RMS for the network and scaled back afterwards. |
| `nn.flow`, `nn.flow_tcnn_feq`, `nn.flow_tcnn_fcnn`, `nn.flow_ftf` | The flows: which networks each uses and its forward pass. A new flow is one case in `nn.flow` plus one forward function. |
| `nn.to_planes` | LLRs in transmit order → bit planes on the grid, so the conventional demapper's output uses the FCNN's loss and labels. |
| `nn.in_training` | Frozen networks run in prediction mode, so their batch-norm statistics stay fixed. |
| `train_receiver.m` | Trains any flow. `'Flow'` picks the chain, `'InitFrom'` loads networks from a saved model, `'Freeze'` keeps named networks fixed, `'Resume'` continues an interrupted run and `'FindLearningRate'` runs a learning-rate range test. It saves `model.Flow`, `model.Nets` (one field per network) and `model.Trainable`, plus `models/<Name>_last.mat` after every epoch. |
| `nn.learning_rate`, `nn.clip_gradients` | The learning-rate schedule (warmup, hold, linear or cosine decay) and per-network gradient clipping used by `nn.train`. |
| `nn.receive` | Runs any saved model's flow. |

**Training the PA experiments** (shared settings from `experiments/pa_common.m`):

```matlab
addpath experiments
args = pa_common;
train_receiver(args{:}, 'Flow', 'fcnn', 'Name', 'fcnn_pa')           % FCNN alone
train_receiver(args{:}, 'Flow', 'tcnn_feq', 'Name', 'tcnn_feq')      % TCNN + conventional
train_receiver(args{:}, 'Flow', 'tcnn_fcnn', 'Name', 'tcnn_fcnn', ...
    'InitFrom', 'models/fcnn_pa.mat', 'Freeze', {'fcnn'})            % TCNN in front of the trained FCNN
```

To fine-tune both networks of the hybrid afterwards, run `train_receiver(args{:},
'Flow','tcnn_fcnn', 'Name','tcnn_fcnn_ft', 'InitFrom','models/tcnn_fcnn.mat',
'LearningRate',1e-4)`. Without `Freeze`, both networks learn.

The TCNN works on 7680 samples × 2 antennas per slot. Expect an epoch to take about as long as
the FCNN's, because slot simulation dominates.

#### Learning-rate schedule, gradient clipping and Resume: `experiments/pa_schedule.m`

`pa_schedule` is `pa_common` plus a training recipe. Train every receiver you compare with it:

| Setting | `pa_schedule` | Default (`nn_config`) |
|---|---|---|
| `MiniBatchSize` | 10 (100 Adam steps per 1000-slot epoch, 15 000 in all) | 4 |
| `LearningRate` (peak) | 3e-3 | 1e-3 |
| `LearningRateSchedule` | `'linear'` | `'constant'` |
| `WarmupSteps` | 200 (2 epochs): the rate rises linearly from 0 | 0 |
| `DecayStart` | 0: the decay starts right after the warmup | 0 |
| `FinalLearningRate` | 2e-3, reached at the last step | 0 |
| `GradientClip` | 20 per network | `Inf` (off) |

`nn.learning_rate` computes the rate at every step: warmup to `LearningRate` over `WarmupSteps`,
held until `DecayStart` (a fraction of all steps; never before the warmup ends), then `'linear'`
or `'cosine'` decay to `FinalLearningRate` at the last step. The defaults keep the old constant
rate.

`GradientClip` caps each trained network's gradient norm at every step (`nn.clip_gradients`),
keeping its direction, so one bad batch cannot wreck a network at a high learning rate. It is
per network, so the LLR calibration's large gradients (scaled by its gain of 50) never shrink
the TCNN's step. Measured at the start of training with batch 10: FCNN about 2; TCNN 6–10 with
spikes up to 250; calibration 3–18 with spikes up to 180. A clip of 20 only cuts the spikes. The
epoch log (`grad`) and `history.GradientNorm` (one column per network) show the median norms;
keep the clip at 3–5× the typical value.

```matlab
addpath experiments
args = pa_schedule;
% 1. Learning-rate range test (a few minutes): rates rise from 1e-5 to 1e-1 over 300 steps
train_receiver(args{:}, 'Flow','tcnn_feq', 'Name','tcnn_feq_s', 'FindLearningRate',true)
% 2. Train (override the peak if the test suggests another, e.g. 'LearningRate',2e-3)
train_receiver(args{:}, 'Flow','tcnn_feq', 'Name','tcnn_feq_s')
train_receiver(args{:}, 'Flow','fcnn', 'Name','fcnn_pa_s')        % the same recipe for the FCNN
% 3. Continue after an interruption: the same call plus 'Resume',true
train_receiver(args{:}, 'Flow','tcnn_feq', 'Name','tcnn_feq_s', 'Resume',true)
```

**Range test.** `'FindLearningRate',true` trains for `RangeTestSteps` (300) steps while the rate
rises exponentially over `RangeTestRates` ([1e-5 1e-1]), stops once the smoothed loss reaches 4×
its minimum, and suggests a tenth of the rate at the lowest loss. The steps are saved in
`models/<Name>_lr_range.mat` (`steps.LearningRate`, `steps.SmoothedLoss`; plot one against the
other on a log x-axis). Nothing else is saved. For `tcnn_feq`, keep the peak at or below about
3e-3 even if the test suggests more: gradients through the LMMSE equalizer spike in deep fades.

**Three ways to start:**

| | Networks | Adam state and schedule | Use for |
|---|---|---|---|
| default | new, random (the TCNN starts as pass-through) | new | a new run |
| `'InitFrom','models/x.mat'` | loaded from `x.mat` | new (warmup protects the loaded weights) | fine-tuning, new conditions, a new recipe; give it a new `Name` |
| `'Resume',true` | from `models/<Name>_last.mat` | continued: Adam moments, step count, schedule position, history, best checkpoint, random state | an interrupted run |

`Resume` continues exactly where the run stopped: 2 epochs followed by a resume to 3 gives the
same weights as 3 epochs in one go (checked in `test_train`). The next epoch's planned slots are
used, so no slot repeats. Give the same settings as the first call: those that define the slots,
the steps or the networks (`MiniBatchSize`, seeds, ranges, widths, ...) are checked. `MaxEpochs`
may grow; with a decaying schedule the decay then stretches from that point, so the rate jumps
up a little. `models/<Name>.mat` stays the best-validation model, as before.

#### Relation to the paper

The flows follow Pihlajasalo et al., "Deep Learning OFDM Receivers for Improved Power
Efficiency and Coverage", *IEEE Trans. Wireless Commun.* 22(8), 2023:

| Here | Paper |
|---|---|
| `'fcnn'` | DeepRx (FCNN), Sec. III-B, the benchmark |
| `'tcnn_feq'` | **TCNN/FEQ**, Sec. III-C: time-domain CNN + conventional frequency-domain receiver, trained end to end through the untrainable FFT and receiver blocks |
| `'tcnn_fcnn'` | **HybridDeepRx (TCNN/FCNN)**, Sec. III-D: the same TCNN and the DeepRx CNN, joined by the FFT, trained jointly |
| `'ftf'` | **Frequency-domain HybridDeepRx (FTF-CNN)**, Sec. III-E: DeepRx blocks → parallel IFFTs → pseudo-time blocks → parallel FFTs → DeepRx blocks, all after the receiver FFT (see below) |

The conventional receiver (the FEQ) is the paper's: LS at the DMRS, linear interpolation,
LMMSE equalization. The paper uses two DMRS symbols per slot, this project one.

The TCNN (`nn.tcnn`) uses the paper's five pre-activation blocks with widths 64-128-256-128-64,
but not its layout. The paper runs a depthwise-separable 3×3 network over each antenna's slot cut
into OFDM symbols (sample × symbol). `nn.tcnn` instead treats the whole slot as one 1-D signal
with both antennas as channels, and uses dilated 3-tap convolutions along time (dilations 1-2-4-2-1,
about 42 samples ≈ 5.5 µs of view, depth multiplier 2). It also outputs input + correction, with the
correction starting at zero, so an untrained TCNN passes the samples through. That gives about
0.48 M parameters (the paper: about 0.28 M).

The paper also trains differently: TDL-B/C/D for training and TDL-A/E for validation,
back-off −1 to 7 dB, a 17th-order polynomial fitted to a measured PA, batch 20, and learning
rate 1e-3 with an 800-step warm-up and linear decay after 30% of training. The paper says "240
to 300 thousand epochs" on 105 000 training TTIs; that is only possible if an epoch means one
batch of 20 (about 5–6 M TTIs processed, ≈ 50 passes over its data). This project trains on
fresh slots every epoch instead: 150 000 different TTIs in a default run. `pa_common` keeps
this project's link: [1 −1/3] PA, back-off 0–10 dB, TDL-C with randomized Doppler and delay
spread, batch 4, constant learning rate; `pa_schedule` adds batch 10 and a warmup-plus-decay
schedule. The PA bank (30 training and 10 validation PAs, dither proportional to each
coefficient, rising and concave check) follows the paper.

#### Frequency-domain hybrid (FTF-CNN): `'Flow','ftf'`

The paper's main proposal (Sec. III-E, Fig. 2(b)). The front end is the ordinary receiver's
(sync, CP removal, FFT); every learned layer runs after the FFT, on the DeepRx features, but
part of the network works in a *pseudo-time* domain:

```
rx.sync → rx.ofdm_demod → deeprx_features → f1: input conv + j blocks          F × N   (frequency)
            → parallel IFFTs (channel pairs → complex, per OFDM symbol)
            → t: k blocks                                                      M × N   (pseudo-time)
            → parallel FFTs back to the occupied subcarriers
            → f2: l blocks + 3×3 output conv → bit logits                      F × N   (frequency)
```

The idea: f1 removes the channel, t undoes the PA in (pseudo-)time, where it acted, and f2 maps
symbols to bits — the impairments are undone in the reverse order of the transmitter.

| | Paper | Here (`nn.ftf_cnn`, `nn.flow_ftf`) |
|---|---|---|
| Blocks | 11 pre-activation ResNet blocks, as DeepRx (Table III: 654 k parameters) | the FCNN's 11 blocks and widths (`nn.fcnn_layout`, built by `nn.resnet_blocks`), so **exactly the FCNN's 1,234,436 parameters**: f1 177,856 + t 786,432 + f2 270,148 |
| Split j / k / l | not given | `FTFSplit` [4 3 4]: f1 64-64-128-128, t 256-256-256, f2 128-128-64-64 |
| Transforms | Nj/2 parallel IFFTs over subcarriers per symbol, real/imaginary channel pairs, no bottleneck (Fig. 3); size not the primary FFT's | `nn.pair_transform` with `nn.pseudo_time_basis`: channels 1..C/2 real, C/2+1..C imaginary; the occupied subcarriers in their bins of an M-point IFFT, scaled to keep the energy (A′A = I, so the FFT back is A′ and exact) |
| Pseudo-time size M | — | `FTFTransformSize`, default the link's Nfft (512): the transmitter's own sample grid. For a transmit grid column X, A·X/√Nfft *is* that symbol's waveform without CP (checked in `test_ftf`), so t sees the samples the PA distorted. 312 (= F, critically sampled) is the cheaper option |
| Pseudo-time convolutions | 3×3 | the same separable 3×3, dilations `FTFTimeDilations` (default 1: the PA has no memory, and 300 ns is 2–3 samples). Zero padding at the ends of each symbol where the transform is circular |
| Networks | one model | three `dlnetwork`s (`ftf_f1`, `ftf_t`, `ftf_f2`) joined by the fixed transforms, so `nn.train` logs and clips each one's gradient |

It trains from scratch (no pass-through start, no LLR calibration: f2 outputs logits as the
FCNN does). `Widths`/`Dilations` apply to it as to the FCNN, with `sum(FTFSplit)` blocks.

```matlab
addpath experiments
args = pa_schedule;
train_receiver(args{:}, 'Flow','ftf', 'Name','ftf', 'FindLearningRate',true)   % optional, a few minutes
train_receiver(args{:}, 'Flow','ftf', 'Name','ftf')                            % ≈ 2–2.5 h on an RTX 5070
compare_snr({'ftf','fcnn_pa','tcnn_feq_s'}, 'Link',{'PABackoffdB',3}, 'Name','snr_ftf_bo3')
```

Measured with `pa_schedule` on the RTX 5070: about 5 s per 100 training slots, so ≈ 55 s per
1000-slot epoch, and 1.4 GB of GPU memory. Gradient norms at the start are 0.3–0.8 per network,
so `GradientClip` 20 never acts. `fcnn_pa` was trained with `pa_common` (batch 4, constant
1e-3), not `pa_schedule`, so part of any difference may be the recipe; `fcnn_pa_s` would
separate the two.

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
only up to its latest best epoch; `plot_history({'ftf_last'})` reads the resume checkpoint
instead, which has every epoch so far.

Validation sets differ between experiments, e.g. `fcnn` has interferers and `fcnn_pa` has
PAs, so compare their curves for **shape**, not level. Use `compare_snr` to compare receivers
on the same slots.

## Comparing receivers: `compare_snr`

`compare_snr` runs any trained models from `train_receiver` (any flow)
side by side with the paper's two LMMSE baselines: LS-estimated channel ('LMMSE') and known channel ('LMMSE known H'). All receivers see **the same
slots**, and every point gives three numbers: uncoded BER (hard decisions before LDPC, the
receiver itself), coded BER (information bits after LDPC decoding) and BLER (blocks whose CRC
fails, what a real link retransmits). With only 30 slots, coded BER usually drops from about
0.1 to exactly 0 within a few dB, because LDPC either decodes a block fully or fails badly.

```matlab
compare_snr({'fcnn','fcnn_pa','tcnn_feq'})                          % SNR 0:4:32 dB, 30 slots per point
compare_snr({'fcnn_pa','tcnn_feq'}, 'Link',{'PABackoffdB',0}, 'Name','snr_pa0')   % heavy PA
compare_snr({'fcnn','models/old/fcnn_v1.mat'}, 'Labels',{'new','v1'}, ...
    'SNRs',-4:2:32, 'Slots',100, 'Name','snr_new_vs_v1')
```

- **Models:** a name means `models/<name>.mat`. Anything ending in `.mat` is a file path.
- **Test link:** the **first** model's link (PA, Doppler, delay spread, channel model, SIR, …),
  changed with `'Link',{...}` (link_config names). Put the model whose conditions you want
  first, or set them explicitly.
- **PA compensation baselines:** with a PA on the test link, 'LMMSE smoothed' and 'DPoI' are
  added (`rx.receive` modes `'smoothed'` and `'dpoi'`). `'Baselines'` chooses them: any of
  `'smoothed'`, `'dpoi'`, `'dpoi_knownH'`, `'dpoi_genie'`, or `{}` for none.
- **PA:** with a polynomial PA, slots cycle through the 10 **unseen validation PAs** saved in
  the models, so no model is tested on a PA it trained on. The back-off is the link's
  `PABackoffdB`, default 6 dB.
- **Layout check:** a model trained for a different modulation, number of RBs, antennas or
  DMRS layout is rejected with a message naming the setting.
- **Output:** a line per SNR point while it runs, then `results/<Name>.png` (uncoded BER, coded
  BER and BLER vs SNR; a missing point means zero errors) and `results/<Name>.mat` (the results table, the
  conditions and the options).

Options: `'SNRs'` (0:4:32), `'Slots'` (30), `'Seed'` (5001, the first slot seed), `'Link'`,
`'UsePABank'` (true), `'Baselines'`, `'Labels'`, `'Name'` ('snr_sweep'). It runs on the CPU, so it can run
next to a GPU training. With 30 slots, BLER moves in steps of 0.033. Use more slots for smooth
BLER curves.

## Tests

```matlab
test_tx          % transmitter checks
test_chan        % channel checks
test_rx          % receiver checks, end to end with LDPC
test_nn          % neural-receiver pipeline checks (about 1 min with a GPU)
test_tcnn        % TCNN flows (CPU only, so it can run beside a GPU training)
test_ftf         % frequency-domain hybrid: transforms, parameters, gradients, training (CPU, ~6 min)
test_train       % training options: schedule, clipping, Resume, range test (CPU only)
test_mppa        % 3GPP memory polynomial PA and 64QAM (see MPPA.md)
test_dpoi        % Digital Post-Inverse baseline and channel smoothing (CPU, ~1 min)
```

`test_chan` checks the channel against physics: Y = H·X for a static channel and growing intercarrier interference with Doppler, average channel power of 1
for TDL and CDL models, measured SIR and SNR equal to the settings, and an unchanged wanted
signal when interference is added.

`test_tx` checks two things:

1. **The LDPC encoder matches MATLAB's `nrULSCH` bit for bit**, for QPSK and 16QAM, code
   rates 0.3–0.8, and a 106-RB slot with 7 code blocks.
2. **Coded loopback over noise:** Tx → OFDM demodulation → AWGN → soft demapping → LDPC
   decoding. Without noise the transport block is recovered exactly. With noise, BLER drops
   from 1 to 0 between 4 and 6 dB SNR (16QAM, rate 0.48; the theoretical limit is
   ≈ 4.4 dB).

`test_rx` checks `rx.ofdm_demod` against `nrOFDMDemodulate` and `rx.demap_llr` against `nrSymbolDemodulate`,
DMRS timing against the true timing (and the BER cost of the difference) for five channel
models at 0/10/30 dB, `dlarray` results, gradients and batches, and the end-to-end SNR sweep
above (BLER from 1 to ~0, known H never worse than LS).

`test_nn` checks the neural pipeline before any long training: an oracle round trip
(labels → perfect logits → LLRs → LDPC decoding gives zero errors, so bit order and sign are
right), the loss gradient, the paper's network (Table I, 1,234,436 parameters) with gradients
reaching the received samples, a short training run on 8 slots in which loss and BER fall
steadily, and the trained `models/fcnn.mat` beating the conventional receiver.

`test_tcnn` checks the TCNN flows before any long training:
- an untrained `tcnn_feq` gives the conventional receiver's LLRs;
- the bit loss reaches every TCNN weight through the fixed conventional receiver;
- a small TCNN learns to undo heavy PA distortion (0 dB back-off) on 8 slots, beating the conventional receiver on
  loss and BER;
- `nn.receive` reproduces the training forward pass;
- an untrained TCNN in front of an FCNN gives that FCNN's LLRs;
- `train_receiver` with `InitFrom` and `Freeze` leaves the frozen FCNN unchanged;
- on mixed conditions, `tcnn_feq` with the LLR calibration lowers the loss without raising the
  BER (without it, the BER rises).

`test_dpoi` checks the PA compensation baseline:
- `rx.delay_estimate` is closer to the true channel than LS with linear interpolation;
- with per-pilot LS the equalized pilots are exact, so there is nothing to fit;
- the frequency-domain implementation equals the paper's eq. (6) computed in time;
- without a PA and noise, DPoI changes nothing;
- with a PA, no noise and the true channel, the EVM falls, and the pilot fit is close to the
  genie fit;
- the noise estimate matches the error left on the data;
- with a PA at 30 dB, DPoI's uncoded BER is below the same receiver without the post-inverse;
- `compare_snr` adds the baselines.

`test_train` checks the training options:
- `nn.learning_rate`: the default is constant; warmup, hold, linear and cosine decay, and the
  range test's rates have the right values;
- `nn.clip_gradients` caps each network separately and keeps the direction;
- `nn.train` stopped by an error after epoch 2 and resumed from its saved state ends with
  exactly the weights and history of an uninterrupted run, with a decaying rate;
- `train_receiver`: 2 epochs + `Resume` to 3 equals 3 epochs, and a changed `MiniBatchSize`
  is refused;
- `'FindLearningRate',true` runs, stops in time and saves its steps.
