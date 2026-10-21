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
| Perplexity (WikiText-2 ↓) | 22.81 | 101.16 | +78.35 degradation |

Benchmark run: `uv run python scripts/benchmark.py --model Qwen/Qwen3-1.7B --ternary-checkpoint out/qwen3-1.7b/ternary.pt`

### Observations

- nsamples=384 and nsamples=256 both OOM'd at 15-18 GB swap on a 5 GB baseline system
- nsamples=128, seqlen=1024 ran stably with peak swap ~6 GB
- Window [0..0] (4 layers) took ~232s/epoch, [0..2] (approaching fill window) ~80s/epoch (smaller activations)
- Training was resumed once successfully via `bash recipes/qwen3-1.7b-mps.sh --resume`
- Output: `out/qwen3-1.7b/ternary.pt` (1.7 GB, 196 QuantLinear modules)

---

## Experiment 2 — MLX Inference Benchmarks (Float vs 4-bit vs Ternary)

**Date:** 2026-10-21
**Status:** ✅ Complete

### Models benchmarked

| Model | How produced | Disk (GB) |
|---|---|---|
| Float BF16 | HuggingFace safetensors | 3.80 |
| MLX 4-bit | `mlx_lm.convert --q-bits 4`, ~2 min, no training | 0.912 |
| MLX Ternary | CAT-Q training + `export_mlx.py` | 0.543 |

### Inference metrics (MLX, peak Metal memory)

| Metric | Float BF16 | MLX 4-bit | MLX Ternary |
|---|---|---|---|
| Disk size (GB) | 3.80 | 0.912 | 0.543 |
| Peak memory (GB) | — | 2.183 | 1.249 |

### Zero-shot accuracy (5-task suite, lm_eval 0.4.7, 0-shot)

Tasks: PIQA, ARC-easy, ARC-challenge, HellaSwag, WinoGrande.
Metric: `acc_norm` where available, `acc` otherwise.

| Task | Float BF16 | MLX 4-bit | MLX Ternary |
|---|---|---|---|
| PIQA | 0.7198 | 0.7089 | 0.6094 |
| ARC-easy | 0.6978 | 0.6620 | 0.4188 |
| ARC-challenge | 0.4309 | 0.4138 | 0.2483 |
| HellaSwag | 0.6048 | 0.5919 | 0.3698 |
| WinoGrande | 0.6109 | 0.5943 | 0.5249 |
| **avg_5** | **0.6128** | **0.5942** | **0.4342** |

### Observations

- MLX 4-bit loses only 1.9 avg_5 points vs float — very small drop for 4× compression.
- MLX Ternary drops 17.9 points vs float. Primary cause: 8× less calibration data (128×1024 vs paper's 512×2048) due to 24 GB memory constraint.
- Memory savings are real: MLX keeps weights packed during inference, never expanding to BF16.
- Raw results: `out/qwen3-1.7b/eval-mlx-float.json`, `eval-mlx-4bit.json`, `eval-mlx-ternary.json`

---

## Experiment 3 — Base-3 Packing: End-to-End Benchmark

**Date:** 2026-10-21  
**Status:** ✅ Complete

### Motivation

Current MLX packing (`pack_mlx_2bit`) stores 16 ternary codes per uint32 = **2 bits/weight**.
Ternary values have 3 states, so the information-theoretic minimum is log2(3) ≈ **1.585 bits/weight**.

### Approach

Base-3 encoding: 3^20 = 3,486,784,401 < 2^32, so **20 codes per uint32** = **1.6 bits/weight**.

Key design decision: pack flat across all groups in each weight matrix (not per group).
Group size is 128 which is not divisible by 20, so per-group packing wastes 12 slots/group
→ 1.75 bpw (12.5% saving). Packing flat across the full weight yields ~1.6 bpw (20% saving).

### Disk size

| Scheme | Total model (GB) | bits/weight |
|---|---|---|
| MLX 2-bit ternary | 0.543 | 2.000 |
| MLX base-3 ternary | 0.501 | 1.600 |

**8% smaller on disk** (20% saving on the ternary weight tensors themselves; non-ternary layers stay 4-bit).
Theoretical minimum: 1.585 bits/weight (we're 1% above ideal).

### Inference benchmark (M5 Pro, Metal, 20-token generation, warm run)

| Model | Peak memory (GB) | Speed (tok/s) |
|---|---|---|
| MLX 4-bit | 1.003 | 133.5 |
| MLX Ternary 2-bit | 0.720 | 146.5 |
| MLX Ternary base-3 | **1.951** | **11.9** |

Test: `tests/test_memory_comparison.py` (`uv run --extra mlx pytest -v -s`)

### Root-cause analysis

Base-3 is 12× slower and uses 2.7× more peak memory than 2-bit despite smaller disk size.

The problem is in `TernaryLinear.__call__`:

```python
def __call__(self, x):
    w = dequantize_base3(self.weight, self.scales)  # materialises full float16 matrix
    return x @ w.T
```

Every forward pass through each of 196 layers allocates a full float16 weight tensor
(e.g. 2048×2048 = 16 MB) before the matmul, then discards it. MLX's built-in 2-bit
`QuantizedLinear` uses a fused dequant+matmul Metal kernel that never stores the full matrix.

### Conclusion

Base-3 packing is **not viable** with a materialise-then-matmul approach.
To make it practical the Metal kernel must be a **fused dequant+matmul** that reads packed
uint32 words and accumulates into the output directly — never materialising the weight matrix.
This is a significantly more complex kernel (requires tile-level matmul logic in MSL).

### Files

- `src/ternary_quench/packing.py`: `pack_mlx_base3`
- `src/ternary_quench/export_mlx.py`: `--base3` flag
- `src/ternary_quench/decode_base3.py`: Metal kernel + `TernaryLinear` + `patch_base3`
- `tests/test_decode_base3.py`: roundtrip correctness (3 passed)
- `tests/test_base3_inference.py`: patching + forward pass (4 passed)
- `tests/test_memory_comparison.py`: memory + runtime comparison across all 3 formats

---

## Reference

- Paper: CAT-Q (Reconstruction-based Ternary Quantization)
- Upstream repo: https://github.com/penk/ternary-quench
- Our fork: https://github.com/immortal-dragon-2311/ternary-quench
- Released 1.7B checkpoint: `penkia/catq-qwen3-1.7b-ternary-st-epoch60-boundary-depth28`
  - ternary.pt: 1.86 GB
  - Float PPL: 34.51, Ternary PPL: 38.75
