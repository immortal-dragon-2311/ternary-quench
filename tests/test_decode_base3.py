"""Roundtrip test: pack with numpy, decode with Metal kernel, compare."""

from __future__ import annotations

import numpy as np
import pytest
import mlx.core as mx


def pack_base3(codes: np.ndarray) -> tuple[np.ndarray, int]:
    """Numpy reference packer — same logic as the planned export_mlx.py --base3 path."""
    flat = codes.ravel()
    n = len(flat)
    pad = (20 - n % 20) % 20
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
    c = flat.astype(np.uint64).reshape(-1, 20)
    powers = np.array([3 ** i for i in range(20)], dtype=np.uint64)
    packed = (c * powers).sum(axis=1).astype(np.uint32)
    return packed, n


def test_roundtrip_small():
    """2×128 weight matrix — exactly one word per group, clean alignment."""
    rng = np.random.default_rng(0)
    out_features, in_features = 2, 128
    codes  = rng.integers(-1, 2, size=(out_features, in_features), dtype=np.int8)
    scales = rng.random((out_features, in_features // 128)).astype(np.float16)

    # Shift codes {-1,0,+1} -> {0,1,2} for packing
    shifted = (codes.astype(np.int16) + 1).astype(np.uint8)
    packed_np, _ = pack_base3(shifted)

    from ternary_quench.decode_base3 import dequantize_base3
    result = dequantize_base3(
        mx.array(packed_np, dtype=mx.uint32),
        mx.array(scales,    dtype=mx.float16),
    )
    mx.eval(result)
    result_np = np.array(result)

    expected = (codes.astype(np.float32) * scales.repeat(in_features, axis=1)).astype(np.float16)
    np.testing.assert_array_equal(result_np, expected)


def test_roundtrip_unaligned():
    """4×256 weight matrix — 1024 weights, 1024 % 20 != 0, tests padding path."""
    rng = np.random.default_rng(1)
    out_features, in_features = 4, 256
    codes  = rng.integers(-1, 2, size=(out_features, in_features), dtype=np.int8)
    scales = rng.random((out_features, in_features // 128)).astype(np.float16)

    shifted = (codes.astype(np.int16) + 1).astype(np.uint8)
    packed_np, _ = pack_base3(shifted)

    from ternary_quench.decode_base3 import dequantize_base3
    result = dequantize_base3(
        mx.array(packed_np, dtype=mx.uint32),
        mx.array(scales,    dtype=mx.float16),
    )
    mx.eval(result)
    result_np = np.array(result)

    expected = (codes.astype(np.float32) * scales.repeat(128, axis=1)).astype(np.float16)
    np.testing.assert_array_equal(result_np, expected)


def test_zero_weights():
    """All-zero codes should produce all-zero output regardless of scale."""
    out_features, in_features = 2, 128
    codes  = np.zeros((out_features, in_features), dtype=np.int8)
    scales = np.ones((out_features, in_features // 128), dtype=np.float16)

    shifted = (codes.astype(np.int16) + 1).astype(np.uint8)  # all 1s
    packed_np, _ = pack_base3(shifted)

    from ternary_quench.decode_base3 import dequantize_base3
    result = dequantize_base3(
        mx.array(packed_np, dtype=mx.uint32),
        mx.array(scales,    dtype=mx.float16),
    )
    mx.eval(result)
    np.testing.assert_array_equal(np.array(result), np.zeros_like(np.array(result)))
