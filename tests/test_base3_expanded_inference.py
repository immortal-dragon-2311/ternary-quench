"""Verify expanded base-3 model gives identical outputs to the original 2-bit model."""

from __future__ import annotations

import mlx.core as mx
import mlx_lm
import numpy as np
import pytest

ORIG_DIR     = "out/qwen3-1.7b/mlx-ternary"
EXPANDED_DIR = "out/qwen3-1.7b/mlx-ternary-base3-expanded"
PROMPT       = "The capital of France is"
MAX_TOKENS   = 20


@pytest.fixture(scope="module")
def orig_model():
    model, tokenizer = mlx_lm.load(ORIG_DIR)
    return model, tokenizer


@pytest.fixture(scope="module")
def expanded_model():
    model, tokenizer = mlx_lm.load(EXPANDED_DIR)
    return model, tokenizer


def test_outputs_match(orig_model, expanded_model):
    orig_out = mlx_lm.generate(
        *orig_model, prompt=PROMPT, max_tokens=MAX_TOKENS, verbose=False
    )
    exp_out = mlx_lm.generate(
        *expanded_model, prompt=PROMPT, max_tokens=MAX_TOKENS, verbose=False
    )
    assert orig_out == exp_out, (
        f"Output mismatch:\n  orig:     {orig_out!r}\n  expanded: {exp_out!r}"
    )
    print(f"\nOutput: {orig_out!r}")


def test_peak_memory_similar(orig_model, expanded_model):
    mx.reset_peak_memory()
    mlx_lm.generate(*orig_model, prompt=PROMPT, max_tokens=MAX_TOKENS, verbose=False)
    orig_gb = mx.get_peak_memory() / 1e9

    mx.reset_peak_memory()
    mlx_lm.generate(*expanded_model, prompt=PROMPT, max_tokens=MAX_TOKENS, verbose=False)
    exp_gb = mx.get_peak_memory() / 1e9

    print(f"\nPeak memory — orig: {orig_gb:.3f} GB  expanded: {exp_gb:.3f} GB")
    # Allow 5% tolerance for minor MLX allocator variation
    assert abs(orig_gb - exp_gb) / orig_gb < 0.05, (
        f"Peak memory differs by more than 5%: {orig_gb:.3f} vs {exp_gb:.3f} GB"
    )
