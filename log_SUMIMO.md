# SU-MIMO extension log

Working log for extending Mini-DeepRx from 1 Tx x 2 Rx single stream to SU-MIMO
(2 Tx x 2 Rx, two layers), while keeping SISO and SIMO as the one-layer case of the
same code. Newest stage last.

## Plan

| Stage | Content | Status |
|---|---|---|
| 0 | Layer-general shapes; one layer bit-identical to before | **done** (2026-11-12) |
| 1 | Linear link with known channel: per-RE MIMO MMSE detection | **done** (2026-11-25) |
| 2 | Pilot-based reception: multi-port DMRS estimation, interpolation, noise estimate | not started |
| 3 | Neural reception: train with `2*(Nr + L + Nr*L)` inputs and `Q*L` outputs | not started |
| 4 | Coded transmission end to end with two layers | not started |
| 5 | Nonlinear transmission: per-antenna PAs, PA compensation revisited | not started |

## Conventions (fixed in stage 0)

`F` subcarriers, `N` OFDM symbols, `Nr` receive antennas, `Nt` transmit antennas,
`L` layers, `B` slots in a batch, `Q` bits per symbol, `D` data REs per layer.

- **The batch stays in dimension 4, the layer goes in dimension 5.** With one layer the
  fifth dimension is a trailing singleton, so every array is exactly what it was.

  | Array | Size |
  |---|---|
  | Transmit grid `truth.Grid` | `F x N x Nt` |
  | Received grid `Y` | `F x N x Nr x B` (unchanged) |
  | Channel `truth.H`, estimate `H` | `F x N x Nr x B x L` (`truth.H`: `F x N x Nr x 1 x Nt`) |
  | Pilot estimates `Hp` | `P x K x Nr x B x L` |
  | Known pilots `ctx.PilotGrid` | `F x N x 1 x 1 x L` (`F x N x 1 x B x L` in a batch) |
  | PA gain `truth.PAGain` | `1 x 1 x 1 x 1 x Nt`, or `F x 1 x 1 x 1 x Nt` for `mp3gpp` |
  | Network input | `F x N x 2*(Nr + L + Nr*L) x B` |
  | Labels, logits | `F x N x (Q*L) x B` |
  | Equalized symbols `xhat`, `nu` | `(L*D) x B` |
  | Coded bits, LLRs | `(Q*L*D) x B` |

- **Bit and symbol order** follows the 5G layer mapping (TS 38.211 6.3.1.3): symbol
  `L*(i-1)+l` is layer `l` of data RE `i`. Bit planes are layer by layer: plane
  `Q*(l-1)+b` is bit `b` of layer `l`.
- **The receiver is indexed by layers, not transmit antennas.** There is no precoder
  yet, so layer `l` is sent from antenna `l` and `link_config` requires
  `NumLayers == NumTransmitAntennas` (and `NumLayers <= NumReceiveAntennas`).
- **Power:** the whole grid, pilots included, is scaled by `1/sqrt(Nt)`. Total transmit
  power does not depend on `Nt`, so `SNRdB` (Es/N0 per RE per receive antenna) keeps its
  meaning. `ctx.PilotGrid` keeps unit-magnitude pilots, so an estimated channel includes
  the `1/sqrt(Nt)`; `truth.H` is the channel alone, and the data sees
  `PAGain .* H / sqrt(Nt)`.
- **Network features** are ordered `[Y (Nr), Xp (L), Hraw (Nr per layer, layer by layer)]`,
  real parts then imaginary parts. With one layer this is the old ten-channel input for
  `Nr = 2`, so the saved models still fit.

## Stage 0: layer-general shapes (2026-11-12)

Goal: every block accepts the layer dimension, and a one-layer link gives exactly the
same numbers as before. No MIMO detection or multi-port channel estimation yet.

### What changed

Configuration
- `link_config`: new `NumTransmitAntennas` (default 1) and `NumLayers` (default 1, at
  most 4), with the two checks above.
- `rx.context`: new `NumLayers`, `NumTransmitAntennas`; `PilotGrid`, `PilotValues` and
  `PilotWaveform` carry one entry per layer; `DataIndices` are the `D` data REs shared
  by all layers; `PilotMask` marks the pilots of any layer.

Transmitter
- `tx.resources`: PUSCH with `cfg.NumLayers` layers; indices and pilots have one column
  per layer. DMRS ports are the toolbox default `0..L-1`.
- `tx.build_grid`: takes `D x L` symbols, returns an `F x N x L` grid.
- `tx.transmit`: layer mapping, `1/sqrt(Nt)` scaling, one PA per transmit antenna (same
  model and back-off), `PAGain` per antenna along dimension 5.
- `tx.transport_block_size`, `tx.ldpc_encode`, `rx.ldpc_decode`: the layer count is
  passed to `nrTBS` and to LDPC rate matching and recovery (`code.NumLayers`).

Channel
- `chan.fading`: `Nt` transmit antennas for TDL and CDL; returns all `Nr x Nt`
  responses as `F x N x Nr x 1 x Nt`.

Conventional receiver
- `rx.sync`: correlates with each layer's pilot waveform and sums the power over layers
  and antennas.
- `rx.ls_estimate`, `rx.interpolate`, `rx.delay_estimate`: layer dimension carried
  through. `rx.ls_estimate` still **refuses more than one layer**, because ports
  `0..L-1` share their resource elements and are not separated yet (stage 2).
- `rx.equalize`, `rx.dpoi`: documented with the layer-general shapes; both **refuse
  more than one layer** (stage 1 and stage 5).

Neural receiver
- `nn.deeprx_features`: pilots and raw LS channel of every layer as extra input
  channels.
- `nn.labels`, `nn.to_llr`, `nn.to_planes`, `nn.make_data`, `nn.train`: `Q*L` bit
  planes and the per-layer pilot grid.
- New `nn.fcnn_input_size(ctx)`; `train_receiver` uses it and builds networks with
  `Q*L` outputs.
- `compare_snr`: the model/link check also compares `NumTransmitAntennas` and
  `NumLayers` (models saved before these settings count as 1).

Tests and docs
- New `test_layers.m` with `test_layers_reference.mat` (see below).
- `README.md`: intro, the `link_config` table and the test list mention the new
  settings and test.

### How it was verified

All runs on this machine, MATLAB at `D:\MATLAB`.

1. **Bit-identical with one layer.** Before any edit, nine links were run on the
   original code (commit `a49064b`) and every output saved: default SIMO, QPSK + LDPC
   with two DMRS symbols, SISO 64QAM on TDL-A, 4 Rx on CDL-C, an interferer, the
   polynomial PA, the `mp3gpp` PA with all four DPoI modes, DMRS type 2 double symbol,
   and perfect timing at 400 Hz. Also a four-slot `nn.make_data` batch with two pilot
   layouts and the saved models `fcnn`, `ftf`, `tcnn_feq`, `ftf_mppa` through
   `nn.receive`. A second run of the original code reproduced all of it exactly. After
   the refactor: **240 of 240 outputs identical** (`isequaln`, no tolerance), including
   the four models' logits and LLRs.
2. **Permanent regression test.** `test_layers` part 1 repeats the nine links and
   compares exact checksums with `test_layers_reference.mat`. The reference was stored
   by running the test on a clean copy of commit `a49064b`, so it is the old code's
   output, not the new code's. All nine cases pass (15 or 16 outputs each).
3. **Two layers through the parts that exist** (`test_layers` parts 3 and 4, 2 Tx x 2 Rx):
   - shapes as in the table above;
   - layer mapping equals `nrLayerMap` (difference 0);
   - total transmit power, two layers / one layer: 1.0027;
   - received grid equals the sum over layers of `H .* X` on a static noiseless channel
     (relative error 1.6e-3, which checks the layout of `truth.H`);
   - DMRS timing sync on both layers' pilots: at most 2 samples from the true timing
     over 10 slots at 10 dB;
   - labels -> oracle logits -> LLRs -> LDPC decoding: 0 errors and CRC passes for QPSK
     (8064-bit block) and 16QAM (16136-bit block);
   - `nn.deeprx_features` gives `312 x 14 x 16`.
4. **Guards** (`test_layers` part 5): the conventional receiver, `rx.equalize`, and the
   two invalid `link_config` combinations fail with a clear message.
5. **Existing tests.** `test_tx`, `test_chan`, `test_rx`, `test_dpoi`, `test_mppa`:
   all pass.

`test_nn`, `test_ftf`, `test_tcnn` and `test_train` all pass as well (24 checks, 0
failures). They finished after the first stage 0 commit `c776b3c`, on the same code.

### Not done in stage 0, and things to know

- A two-layer link cannot be received yet: there is no MIMO detector (stage 1) and no
  DMRS port separation (stage 2). The transmitter and channel do run with two layers.
- No network has been trained or run with two layers; only the input and label shapes
  were checked.
- The two-layer DMRS uses ports 0 and 1, which share one comb and are separated by a
  code. Stage 2 has to choose between despreading them and moving to ports 0 and 2
  (separate combs, per-RE estimation as now, no data in the pilot symbol's gaps, 3 dB
  pilot boost). `rx.context` currently asserts that all layers' pilots sit on the same
  REs, so ports 0 and 2 need a change there.
- `knownH` mode will need `truth.H / sqrt(Nt)` (and the per-antenna PA gain) in stage 1.
- The PA bank draws one PA per slot; with two transmit antennas both currently get the
  same PA model and coefficients. Stage 5 should decide whether they differ.
- `MIMOCorrelation` is fixed at `'Low'`; the poorly conditioned test channels of the
  draft need it configurable.
- The reference checksums are exact, so another MATLAB release or CPU may change them
  in the last bit. After confirming that is the cause, run `test_layers('update')`.

## Stage 1: MIMO LMMSE detection with the known channel (2026-11-25)

Goal: receive a two-layer slot when the channel is known (`'knownH'`), with the
one-layer link still bit-identical.

### What changed

- `rx.equalize`: with more than one layer, per-RE LMMSE detection. Per data RE,
  `y = H x + n` with `H` `Nr x L`, `A = H'H + sigma2*I`:
  `x_lmmse = A^-1 H' y`, `mu_l = 1 - sigma2*[A^-1]_ll`,
  `xhat_l = x_lmmse_l / mu_l` (unbiased), `nu_l = (1 - mu_l)/mu_l`
  (remaining noise plus other-layer leakage, treated as Gaussian, for the soft
  demapper). Vectorized over REs and the batch with `pagemtimes`/`pageinv`. With one
  layer these formulas are the old MRC ones, and the code takes the old path unchanged
  (still dlarray-safe; the multi-layer path is numeric only).
  Output order `(L*D) x B`, layer fastest, so `rx.demap_llr` gives transmit bit order.
- `rx.receive` `'knownH'`: the channel is `genie.H .* PAGain / sqrt(Nt)` (what the data
  actually sees; the division is skipped for `Nt = 1`). With more than one layer the
  noise power cannot come from `rx.ls_estimate` (the DMRS ports are not separated), so
  it is the pilot residual with the true channel: `mean |Y - sum_l H_l X_l|^2` over the
  pilot REs and antennas (local function `pilotNoise`). `trace.Hp` is empty in that
  case. All other modes still refuse more than one layer (`rx.ls_estimate`, `rx.dpoi`).
- `test_layers`: new part 5 (below); the old guard "rx.equalize refuses two layers" is
  replaced by "DPoI refuses two layers".
- `README.md`: the `NumLayers` row and the test list.

### How it was verified

1. **One layer unchanged.** `test_layers` part 1: all nine stored cases identical
   (15-16 outputs each, exact checksums against the pre-SU-MIMO reference), including the
   `'knownH'` LLRs.
2. **Detector against the textbook formula** (`test_layers` part 5): `rx.equalize`
   versus a per-RE loop with `inv(H'H + s*I)*H'*y`, batch of 2 with different noise
   powers: relative difference 7.6e-15 (2 x 2) and 1.3e-15 (4 x 2). With the second
   layer's channel set to zero, the first layer equals the one-layer equalizer (2e-16).
3. **Noiseless static link**, 2 x 2 16QAM: EVM of the detected symbols -45.9 dB (the
   floor is the ~1e-3 mismatch between `truth.H` and the actual channel seen in stage 0),
   noise estimate 3e-6.
4. **Coded at 20 dB** (TDL-C, 100 Hz): QPSK and 16QAM decode, CRC passes; noise
   estimate / true N0 = 1.14 on that slot.
5. **SNR sweep**, `'knownH'`, LDPC R = 0.5, TDL-C 300 ns, 100 Hz, 2 Rx, 40 slots per
   point (seeds 1-40):

   | Link | TB bits | BLER < 10% from | BER at 10 dB | BER at 20 dB |
   |---|---|---|---|---|
   | 1x2, 1 layer, QPSK | 4032 | 2 dB (0.050), 4 dB (0) | 0.0055 | 0.0001 |
   | 2x2, 2 layers, QPSK | 8064 | 6 dB (0.025) | 0.0530 | 0.0066 |
   | 1x2, 1 layer, 16QAM | 8064 | 8 dB (0) | 0.0451 | 0.0012 |
   | 2x2, 2 layers, 16QAM | 16136 | 14 dB (0.025) | 0.1533 | 0.0304 |

   Full BLER rows (0, 2, ..., 12 dB): 1x2 QPSK 0.225 0.050 0; 2x2 QPSK 1.000 0.875
   0.325 0.025 0; 1x2 16QAM 0.975 0.750 0.450 0.175 0; 2x2 16QAM 1.000 1.000 1.000
   1.000 0.925 0.475 0.175 (0.025 at 14 dB, 0 from 16 dB).
   - Doubling the rate with a second layer costs about 4 dB (QPSK) and 6 dB (16QAM) at
     this SNR definition (total transmit power fixed).
   - At the same 8064-bit block, 2x2 QPSK needs about 1-2 dB less than 1x2 16QAM.
   - The two-layer uncoded BER falls slowly with SNR: with `Nr = L = 2` the LMMSE
     detector has no spare receive diversity per layer.
   - Noise estimate / true N0, averaged over 40 slots: 1.00 up to about 12 dB, rising
     to 1.07 at 24 dB for two layers (1.03 for one layer). At 30 dB on one slot:
     1.12 static and 1.32 at 100 Hz for the two-layer pilot residual, against 1.15 and
     1.18 for the one-layer `rx.ls_estimate`. The floor is the `truth.H` mismatch plus
     Doppler leakage; it only matters at high SNR.
6. **Existing tests**: `test_tx`, `test_chan`, `test_rx`, `test_dpoi`, `test_mppa` all
   pass on this code (64 checks, 0 failures). The neural tests were not rerun: the
   one-layer `rx.equalize` path they use (`nn.flow_tcnn_feq`) is unchanged, and
   `test_layers` part 1 shows the one-layer outputs are bit-identical.

### Not done in stage 1, and things to know

- The pilot-based receiver still refuses two layers: DMRS port separation, multi-port
  interpolation and the noise estimate are stage 2. The `'knownH'` noise estimate uses the
  true channel and is a genie quantity like `H` itself.
- The per-layer `nu` treats the other layer's leakage as Gaussian. A max-log ML or
  sphere detector would do better at high SNR with `Nr = L`; worth a comparison once the
  neural receiver has two layers (stage 3).
- `MIMOCorrelation` is still fixed at `'Low'`; the sweep above is for that setting only.

