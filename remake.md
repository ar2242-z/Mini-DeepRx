# Remake plan: signal-chain structure

Goal: rebuild the code around the link chain (Tx → channel → Rx) so that every
receiver (conventional, TCNN, FCNN/DeepRx, hybrids) shares the same Tx, channel,
decoder and evaluation, and differs only in a pipeline definition.

## Signal chain

```
Tx:    info bits → LDPC encode + rate match → symbol mapping → grid (data + DMRS)
       → IFFT + CP → PA
Chan:  time-domain fading → interference → noise
Rx:    sync (timing) → CP removal + FFT → LS estimate + interpolation
       → equalization / SIMO combining → soft demap (LLRs) → LDPC decode → bits
```

## Layout

```
+tx/    bits  ldpc_encode  map_symbols  build_grid  ofdm_mod  pa
+chan/  fading  interference  awgn
+rx/    sync  ofdm_demod  ls_estimate  ls_raw  interpolate  equalize  demap_llr
        ldpc_decode  true_h
+nn/    tcnn  fcnn_llr  deeprx_features  tcnn_features  train_loop  loss_bce  plan_slots
run_link.m      cfg -> [truth, rxIn, trace]
receive.m       rxIn, pipeline, ctx, params -> llr, trace
evaluate.m      llr, truth -> uncoded BER, coded BER, BLER
make_dataset.m  cfg, N -> shards of (rxIn, truth)
```

- Keep `simo_config` as the single config struct and pass it through.
- Move the duplicated helpers (`pilotStride`, `makePlan`, `batchArray`,
  `transferNetwork`, `lossPerBit`) into `+nn/` or `+rx/`.

## Design rules

1. **Rx blocks are differentiable.** Use only basic array ops (DFT matrix,
   element-wise LS/MMSE, max-log demap) so the same code runs on `double` and
   `dlarray`. No `nrEqualizeMMSE` in the Rx path. This replaces `feqReceiver` in
   `TCNN.m`.
2. **Receiver = list of stages**, each `[s, aux] = stage(s, aux, ctx, params)`:
   ```matlab
   rx.conventional = {@rx.sync,@rx.ofdm_demod,@rx.ls_estimate,@rx.interpolate,@rx.equalize,@rx.demap_llr};
   rx.tcnn_feq     = {@rx.sync,@nn.tcnn,@rx.ofdm_demod,@rx.ls_estimate,@rx.interpolate,@rx.equalize,@rx.demap_llr};
   rx.fcnn         = {@rx.sync,@rx.ofdm_demod,@rx.ls_raw,@nn.fcnn_llr};
   rx.hybrid       = {@rx.sync,@nn.tcnn,@rx.ofdm_demod,@rx.ls_raw,@nn.fcnn_llr};
   rx.knownH       = {@rx.sync,@rx.ofdm_demod,@rx.true_h,@rx.equalize,@rx.demap_llr};
   ```
3. **Fixed tensor layouts with a batch dimension:**
   time `[samples × Nr × B]`, grid `[subc × sym × Nr × B]`,
   LLR `[subc × sym × bitsPerSym × B]` + data mask. `nn.*` wrappers handle
   `dlarray` formats internally.
4. **LLR convention:** `LLR = log P(b=0)/P(b=1)` (what `nrLDPCDecode` expects).
   Convert network logits at the `receive.m` boundary.
5. **Truth is separate.** `truth` (bits, true H, noise) goes only to `evaluate.m`
   and the explicit `knownH` path, never to practical receivers.
6. **Checkpoints = `trace` struct** returned by each stage (bits, grid, pre/post-PA
   waveform, rx waveform, FFT output, H estimate, equalized symbols, LLRs,
   decoded bits/CRC). Saving to disk is the caller's choice.
7. **Switches:** `cfg.Coding = 'none'|'ldpc'`, `cfg.PAModel` independent.
   LDPC via `nrLDPCEncode` + `nrRateMatchLDPC` (and inverses).
8. **RNG:** one `RandStream` substream per random block (bits, channel,
   interference, noise) so reordering code doesn't change results.

## Training

- One generic loop: `params` is a struct of `dlnetwork`s (`params.tcnn`,
  `params.fcnn`), each optionally frozen; loss = masked BCE on LLRs of coded
  bits. Covers single nets, end-to-end hybrids, and staged training
  (pretrain → freeze → train next → fine-tune).
- LDPC is evaluation only; no training through the decoder.
- Train/test from `make_dataset` shards so all receivers see identical data.

## Order of work

1. **Freeze a reference:** fixed-seed outputs of current `mini_simo_channel`,
   `decode_simo_ls`, and a short `TCNN` validation (X, Y, BERs, LLRs).
2. **Extract Tx / channel / Rx, uncoded.** Match the reference exactly
   (same random draw order), then switch to per-block substreams and re-freeze.
3. **Merge `feqReceiver` into `+rx`.** Check double vs `dlarray` give the same
   LLRs and TCNN validation BER is unchanged.
4. **Pipeline + generic training loop.** Port `network.m` (FCNN) and `TCNN.m`
   onto it; reproduce their results.
5. **Add LDPC,** then sweep coding × PA × receiver.
6. **Hybrids** (TCNN+FCNN, NN channel estimator, NN demapper) as new pipelines.

Out of scope for now: iterative receivers with decoder feedback.
