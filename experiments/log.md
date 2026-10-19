# Experiment Log — Ternary Quench

## Goal
Quantize Qwen3-1.7B to ternary weights ({-1, 0, +1} × per-group scale) using CAT-Q,
then benchmark perplexity, disk size, and memory before/after quantization.
Longer term: improve packing efficiency beyond the current int8 storage.

---

## Hardware

| Property | Value |
|---|---|
| Machine | MacBook Pro, M5 Pro |
| Unified memory | 24 GB |
| PyTorch device | `mps` |
| OS | macOS |

---

## Memory Budget Analysis

Fixed overhead per run:

| Component | GB |
|---|---|
| Model weights (bfloat16) | 3.17 |
| Active window FP32 (4 layers) | 1.00 |
| LoRA + optimizer + OS overhead | 0.60 |
| **Total fixed** | **4.77** |

Available for activation cache: **19.23 GB**

Activation cache = `nsamples × seqlen × hidden_size(2048) × 4 bytes × 3 streams`

Paper params (512 × 2048) need exactly 24.00 GB — zero headroom, will crash on MPS.

Chosen params: **nsamples=384, seqlen=2048** → 18.00 GB cache, ~5.2 GB headroom.

---

## Experiment 1 — Full Training Run on MPS

**Date:** 2026-10-19  
**Status:** 🔄 In progress

### Params

| Param | Value | Paper value | Note |
|---|---|---|---|
| model | Qwen/Qwen3-1.7B | same | |
| nsamples | 384 | 512 | reduced to fit 24GB |
| seqlen | 2048 | 2048 | full context |
| epochs | 60 | 60 | |
| batch-size | 1 | 9 | MPS constraint |
| num-layer | 4 | 4 | |
| sliding-layer | 2 | 2 | |
| lora-r | 64 | 64 | |
| lora-lr | 0.0003 | 0.0003 | |
| factor-lr | 0.0015 | 0.0015 | |
| device | mps | cuda | Apple Silicon |
| cpu-offload | off | on | requires CUDA |
| amp-dtype | none | bfloat16 | requires CUDA |
| seed | 2 | 2 | |

### Command
```bash
bash recipes/qwen3-1.7b-mps.sh
```

### Results
<!-- Fill in after training completes -->

| Metric | Float BF16 | Ternary |
|---|---|---|
| Disk size (GB) | — | — |
| GPU peak memory (GB) | — | — |
| CPU peak memory (GB) | — | — |
| Perplexity (WikiText-2) | — | — |

### Observations
<!-- Fill in during / after run -->

---

## Experiment 2 — Packing Improvement (Planned)

Current storage: int8 (8 bits/weight), but ternary values only need ~1.585 bits.
Plan: pack 5 ternary values into 1 byte (base-3 encoding, 1.6 bits/weight).
Compare vs current int8 and vs lossless compression (zstd).

---

## Reference

- Paper: CAT-Q (Reconstruction-based Ternary Quantization)
- Upstream repo: https://github.com/penk/ternary-quench
- Our fork: https://github.com/immortal-dragon-2311/ternary-quench
- Released 1.7B checkpoint: `penkia/catq-qwen3-1.7b-ternary-st-epoch60-boundary-depth28`
  - ternary.pt: 1.86 GB
  - Float PPL: 34.51, Ternary PPL: 38.75
