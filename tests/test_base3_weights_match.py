"""Verify that base-3 → expanded weights are bit-exact vs the original 2-bit model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.torch import load_file

ORIG_DIR     = Path("out/qwen3-1.7b/mlx-ternary")
EXPANDED_DIR = Path("out/qwen3-1.7b/mlx-ternary-base3-expanded")


@pytest.fixture(scope="module")
def orig_tensors():
    index = json.loads((ORIG_DIR / "model.safetensors.index.json").read_text())
    out = {}
    for shard in sorted(set(index["weight_map"].values())):
        out.update(load_file(ORIG_DIR / shard))
    return out


@pytest.fixture(scope="module")
def expanded_tensors():
    index = json.loads((EXPANDED_DIR / "model.safetensors.index.json").read_text())
    out = {}
    for shard in sorted(set(index["weight_map"].values())):
        out.update(load_file(EXPANDED_DIR / shard))
    return out


@pytest.fixture(scope="module")
def ternary_layer_names():
    # original 2-bit model has no ternary_base3_layers; read from base-3 config
    base3_config = json.loads(
        (Path("out/qwen3-1.7b/mlx-ternary-base3") / "config.json").read_text()
    )
    return base3_config["ternary_base3_layers"]


def test_all_ternary_weights_match(orig_tensors, expanded_tensors, ternary_layer_names):
    mismatches = []
    for stem in ternary_layer_names:
        key = f"{stem}.weight"
        orig = orig_tensors[key].numpy()
        exp  = expanded_tensors[key].numpy()
        if not np.array_equal(orig, exp):
            mismatches.append((key, orig.shape, exp.shape,
                               int((orig != exp).sum()), int(orig.size)))
    if mismatches:
        lines = [f"  {k}: shape {s1} vs {s2}, {nd}/{tot} words differ"
                 for k, s1, s2, nd, tot in mismatches]
        pytest.fail(f"{len(mismatches)} ternary layers have weight mismatches:\n" +
                    "\n".join(lines))


def test_all_scales_match(orig_tensors, expanded_tensors, ternary_layer_names):
    for stem in ternary_layer_names:
        key = f"{stem}.scales"
        np.testing.assert_array_equal(
            orig_tensors[key].numpy(),
            expanded_tensors[key].numpy(),
            err_msg=f"scales mismatch in {stem}",
        )


def test_non_ternary_tensors_unchanged(orig_tensors, expanded_tensors, ternary_layer_names):
    import torch
    ternary_keys = {f"{s}.{t}" for s in ternary_layer_names for t in ("weight", "scales", "biases")}
    mismatches = []
    for key in orig_tensors:
        if key in ternary_keys:
            continue
        if not torch.equal(orig_tensors[key], expanded_tensors[key]):
            mismatches.append(key)
    if mismatches:
        pytest.fail(f"non-ternary tensors changed: {mismatches}")
