#!/usr/bin/env bash
# Full training run on Apple Silicon (MPS, 24 GB unified memory).
# No --cpu-offload (requires CUDA). Activations stay in unified memory.
set -euo pipefail

PYTHONUNBUFFERED=1 uv run ternary-quench \
  --model Qwen/Qwen3-1.7B \
  --out out/qwen3-1.7b/ternary.pt \
  --checkpoint-dir out/qwen3-1.7b/checkpoints \
  --nsamples 384 \
  --seqlen 2048 \
  --epochs 60 \
  --batch-size 1 \
  --num-layer 4 \
  --sliding-layer 2 \
  --fill-window-size 4 \
  --group-size 128 \
  --lora-r 64 \
  --lora-lr 0.0003 \
  --factor-lr 0.0015 \
  --lora-wd 0 \
  --factor-wd 0 \
  --lr-schedule linear \
  --huber-loss-max 1.0 \
  --grad-clip 1.0 \
  --ste tanh \
  --s0 30 \
  --progressive-ratio 0.8 \
  --hard-gradient-mode boundary \
  --calib c4 \
  --seed 2 \
  --device mps
