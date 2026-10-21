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

Chosen params: **nsamples=128, seqlen=1024** → 2.00 GB cache, ~17 GB headroom.

nsamples=384 and nsamples=256 were tried first but caused 15-18 GB swap due to ~5 GB
system baseline. Reduced to 128×1024 for stable training.

---

## Experiment 1 — Full Training Run on MPS

**Date:** 2026-10-19 → 2026-10-20  
**Status:** ✅ Complete

### Params

| Param | Value | Paper value | Note |
|---|---|---|---|
| model | Qwen/Qwen3-1.7B | same | |
| nsamples | 128 | 512 | reduced to fit 24GB |
| seqlen | 1024 | 2048 | reduced to fit 24GB |
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

Training completed successfully overnight (19/19 rounds, 196 quantized linears).

| Metric | Float BF16 | Ternary | Notes |
|---|---|---|---|
| Disk size (GB) | 3.80 | 1.74 | 2.19× smaller |
| GPU peak memory (GB) | 3.22 | 3.22 | same (weights reconstructed to BF16 in-memory) |
| CPU peak memory (GB) | 0.62 | 1.51 | higher due to loading ternary.pt alongside model |
| Perplexity (WikiText-2 ↓) | 22.81 | 101.16 | fixed-stride rolling window via scripts/benchmark.py |

### Observations

- nsamples=384 and nsamples=256 both OOM'd at 15-18 GB swap on a 5 GB baseline system
- nsamples=128, seqlen=1024 ran stably with peak swap ~6 GB
- Window [0..0] (4 layers) took ~232s/epoch, [0..2] (approaching fill window) ~80s/epoch (smaller activations)
- Training was resumed once successfully via `bash recipes/qwen3-1.7b-mps.sh --resume`
- Output: `out/qwen3-1.7b/ternary.pt` (1.7 GB, 196 QuantLinear modules)

---

## Experiment 2 — MLX Inference Benchmarks

**Date:** 2026-10-21  
**Status:** ✅ Complete

### Models

| Model | How produced | Disk (GB) |
|---|---|---|
| Float BF16 | HuggingFace safetensors | 3.80 |
| MLX 4-bit | `mlx_lm.convert --q-bits 4`, ~2 min, no training | 0.912 |
| MLX Ternary 2-bit | CAT-Q training + `export_mlx.py` | 0.543 |
| MLX Ternary base-3 | `export_mlx.py --base3` + `expand_base3` at load | 0.501 |

### Inference (M5 Pro, Metal, 20-token generation, warm run)

| Model | Disk (GB) | Peak memory (GB) | Speed (tok/s) |
|---|---|---|---|
| Float BF16 | 3.80 | — | — |
| MLX 4-bit | 0.912 | 1.004 | 126.4 |
| MLX Ternary 2-bit | 0.543 | 0.720 | 147.7 |
| MLX Ternary base-3 (expanded) | 0.501 | 0.720 | 143.5 |

Test: `tests/test_memory_comparison.py`

### WikiText-2 perplexity (lm_eval 0.4.7, document-level rolling loglikelihood, batch_size=1)

| Model | Word perplexity ↓ |
|---|---|
| Float BF16 | — (pending) |
| MLX 4-bit | 22.92 |
| MLX Ternary 2-bit | 334.44 |
| MLX Ternary base-3 (expanded) | 334.44 (bit-exact with 2-bit) |

Note: Experiment 1 used a different method (fixed-stride rolling window, MPS) so those
numbers (Float=22.81, Ternary=101.16) are not directly comparable to lm_eval above.
Raw: `out/qwen3-1.7b/eval-mlx-{4bit,ternary,ternary-base3-expanded}-ppl.json`

### Zero-shot accuracy (lm_eval 0.4.7, 0-shot)

Tasks: PIQA, ARC-easy, ARC-challenge, HellaSwag, WinoGrande.
Metric: `acc_norm` where available, `acc` otherwise.

| Task | Float BF16 | MLX 4-bit | MLX Ternary 2-bit | MLX Ternary base-3 |
|---|---|---|---|---|
| PIQA | 0.7198 | 0.7089 | 0.6094 | 0.6094 (identical) |
| ARC-easy | 0.6978 | 0.6620 | 0.4188 | 0.4188 (identical) |
| ARC-challenge | 0.4309 | 0.4138 | 0.2483 | 0.2483 (identical) |
| HellaSwag | 0.6048 | 0.5919 | 0.3698 | 0.3698 (identical) |
| WinoGrande | 0.6109 | 0.5943 | 0.5249 | 0.5249 (identical) |
| **avg_5** | **0.6128** | **0.5942** | **0.4342** | **0.4342 (identical)** |

Base-3 expanded is bit-exact with 2-bit — confirmed by weight tensor comparison
(`tests/test_base3_weights_match.py`, 3 passed) and identical forward pass outputs
(`tests/test_base3_expanded_inference.py`, 2 passed).
Raw: `out/qwen3-1.7b/eval-mlx-{float,4bit,ternary}.json`

### Observations

- MLX 4-bit loses only 1.9 avg_5 points vs float — near-lossless at 4× compression.
- MLX Ternary drops 17.9 avg_5 points vs float. Primary cause: 8× less calibration
  data (128×1024 vs paper's 512×2048) due to 24 GB memory constraint on M5 Pro.
- Base-3 packing saves 8% on disk (20% on ternary weight tensors; scales and
  non-ternary layers unchanged) with zero inference penalty after one-time expansion.

---

## Experiment 3 — Base-3 Storage Packing

**Date:** 2026-10-21  
**Status:** ✅ Complete

### Motivation

`pack_mlx_2bit` stores 16 ternary codes per uint32 = **2 bits/weight**.
Information-theoretic minimum for 3-state values: log2(3) ≈ **1.585 bits/weight**.

### Approach

3^20 = 3,486,784,401 < 2^32, so 20 codes fit per uint32 = **1.6 bits/weight** (20% denser).

Packing is done flat across the full weight matrix (not per group) to avoid per-group
padding waste: group_size=128 is not divisible by 20, so per-group packing wastes
8 slots/group → 1.75 bpw. Flat packing across the whole matrix → 1.6 bpw.

At load time, `ternary-quench-expand-base3` unpacks base-3 → standard 2-bit MLX format.
Inference runs through MLX's built-in `QuantizedLinear` — no custom kernel, no changes.

### Disk size

| Scheme | Total model (GB) | Weight tensors only | bits/weight |
|---|---|---|---|
| MLX 2-bit ternary | 0.543 | ~328 MB | 2.000 |
| MLX base-3 ternary | 0.501 | ~263 MB | 1.600 |

8% smaller overall (20% on weight tensors; scales, embeddings, config unchanged).
Theoretical minimum: 1.585 bpw — we are 1% above ideal.

### Pipeline

```
export_mlx.py --base3   →  base-3 model on disk (0.501 GB)
                               ↓  ternary-quench-expand-base3 (~1 sec, CPU)
                           standard 2-bit MLX model (0.543 GB in memory)
                               ↓  mlx_lm.load()
                           inference — identical to native 2-bit
```

### Tests

| Test file | What it checks | Result |
|---|---|---|
| `tests/test_base3_expand.py` | `pack → unpack` numpy roundtrip (6 cases) | ✅ 6 passed |
| `tests/test_base3_weights_match.py` | Expanded weights bit-exact vs original 2-bit (3 checks) | ✅ 3 passed |
| `tests/test_base3_expanded_inference.py` | Identical token output + peak memory (2 checks) | ✅ 2 passed |

### Key files

- `src/ternary_quench/packing.py`: `pack_mlx_base3`, `unpack_mlx_base3`
- `src/ternary_quench/export_mlx.py`: `--base3` flag, writes `ternary_base3_layers` to config.json
- `src/ternary_quench/expand_base3.py`: CLI tool (`ternary-quench-expand-base3`)

---

## Reference

- Paper: CAT-Q (Reconstruction-based Ternary Quantization)
- Upstream repo: https://github.com/penk/ternary-quench
- Our fork: https://github.com/immortal-dragon-2311/ternary-quench
- Released 1.7B checkpoint: `penkia/catq-qwen3-1.7b-ternary-st-epoch60-boundary-depth28`
  - ternary.pt: 1.86 GB
  - Float PPL: 34.51, Ternary PPL: 38.75
