"""Tests for base-3 model patching and end-to-end inference."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

MODEL_DIR = Path("out/qwen3-1.7b/mlx-ternary-base3")


@pytest.fixture(scope="module")
def model_and_tokenizer():
    from mlx_lm.utils import load_model, load_tokenizer
    from ternary_quench.decode_base3 import patch_base3

    model, config = load_model(MODEL_DIR, strict=False)
    tokenizer = load_tokenizer(MODEL_DIR)
    model = patch_base3(model, config["ternary_base3_layers"])
    return model, tokenizer, config


def test_base3_layers_in_config():
    """config.json must list ternary_base3_layers and have 196 entries."""
    config = json.loads((MODEL_DIR / "config.json").read_text())
    assert "ternary_base3_layers" in config
    assert len(config["ternary_base3_layers"]) == 196


def test_patch_replaces_ternary_layers(model_and_tokenizer):
    """After patching, all base-3 layers should be TernaryLinear (1-D uint32 weight)."""
    from ternary_quench.decode_base3 import TernaryLinear

    model, _, config = model_and_tokenizer
    base3_set = set(config["ternary_base3_layers"])
    patched = []
    model.apply_to_modules(
        lambda path, m: patched.append(path) if isinstance(m, TernaryLinear) else None
    )
    assert set(patched) == base3_set, (
        f"Mismatch: {len(patched)} TernaryLinear layers found, expected {len(base3_set)}"
    )


def test_ternary_weight_is_1d_uint32(model_and_tokenizer):
    """TernaryLinear weights must be 1-D uint32 (base-3 packed)."""
    from ternary_quench.decode_base3 import TernaryLinear

    model, _, _ = model_and_tokenizer
    checked = 0
    def _check(path, m):
        nonlocal checked
        if isinstance(m, TernaryLinear):
            assert m.weight.ndim == 1, f"{path}: weight is not 1-D"
            assert m.weight.dtype == mx.uint32, f"{path}: weight dtype is {m.weight.dtype}"
            checked += 1
    model.apply_to_modules(_check)
    assert checked == 196


def test_forward_pass_runs(model_and_tokenizer):
    """A single forward pass should complete without error."""
    import mlx_lm

    model, tokenizer, _ = model_and_tokenizer
    mx.reset_peak_memory()
    response = mlx_lm.generate(
        model, tokenizer,
        prompt="The capital of France is",
        max_tokens=20,
        verbose=False,
    )
    peak_gb = mx.get_peak_memory() / 1e9
    assert len(response) > 0
    print(f"\nGenerated: {response!r}")
    print(f"Peak Metal memory: {peak_gb:.3f} GB")
