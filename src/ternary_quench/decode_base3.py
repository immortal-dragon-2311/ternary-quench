"""Base-3 dequantization kernel and linear layer for MLX inference.

Each uint32 word encodes 20 ternary codes in base-3 (3^20 = 3,486,784,401 < 2^32).
The Metal kernel decodes them back to float16 on the GPU without expanding to a
full float16 buffer in host memory.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as mnn
import numpy as np

_CODES_PER_WORD = 20
_GROUP_SIZE = 128
_THREADGROUP = 256

# ---------------------------------------------------------------------------
# Metal kernel source
# ---------------------------------------------------------------------------
# One thread per uint32 word.  Each word holds up to 20 ternary codes stored
# as digits of a base-3 number: value = sum(code_i * 3^i, i=0..19).
# Decoding: repeatedly take (word % 3) to get the least-significant code,
# then divide by 3 to shift to the next.
#
# Codes {0, 1, 2} map to weights {-scale, 0, +scale} via (code - 1) * scale.
# Scales are float16, one per group of 128 weights, stored row-major alongside
# the weight matrix.

_KERNEL_SOURCE = r"""
    uint tid = thread_position_in_grid.x;
    if (tid >= n_words_buf[0]) return;

    uint32_t word   = packed[tid];
    uint     base   = tid * 20u;
    uint     n_w    = n_weights_buf[0];

    for (uint i = 0u; i < 20u; i++) {
        uint widx = base + i;
        if (widx >= n_w) break;

        uint code = word % 3u;
        word     /= 3u;

        // {0,1,2} -> {-1, 0, +1}
        float val = (float)code - 1.0f;

        uint  gidx = widx / 128u;
        float s    = (float)scales[gidx];

        out[widx] = (T)(val * s);
    }
"""

_kernel: object | None = None


def _get_kernel() -> object:
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="base3_dequantize",
            input_names=["packed", "scales", "n_weights_buf", "n_words_buf"],
            output_names=["out"],
            source=_KERNEL_SOURCE,
        )
    return _kernel


# ---------------------------------------------------------------------------
# Public decode function
# ---------------------------------------------------------------------------

def dequantize_base3(
    packed: mx.array,
    scales: mx.array,
) -> mx.array:
    """Decode base-3 packed weights to a float16 matrix.

    Args:
        packed: uint32 array of shape [num_words], flat across the weight matrix.
        scales: float16 array of shape [out_features, in_features // 128].

    Returns:
        float16 array of shape [out_features, in_features].
    """
    out_features, n_groups = scales.shape
    in_features = n_groups * _GROUP_SIZE
    n_weights = out_features * in_features
    n_words = packed.size

    # Scalar metadata passed as uint32 arrays of length 1
    n_weights_buf = mx.array(np.array([n_weights], dtype=np.uint32))
    n_words_buf   = mx.array(np.array([n_words],   dtype=np.uint32))

    grid_x = math.ceil(n_words / _THREADGROUP) * _THREADGROUP

    outputs = _get_kernel()(
        inputs=[
            packed.flatten(),
            scales.flatten(),
            n_weights_buf,
            n_words_buf,
        ],
        template=[("T", mx.float16)],
        grid=(grid_x, 1, 1),
        threadgroup=(_THREADGROUP, 1, 1),
        output_shapes=[(n_weights,)],
        output_dtypes=[mx.float16],
    )
    return outputs[0].reshape(out_features, in_features)


# ---------------------------------------------------------------------------
# Linear layer
# ---------------------------------------------------------------------------

def patch_base3(model: mnn.Module, base3_layer_names: list[str]) -> mnn.Module:
    """Replace QuantizedLinear layers with TernaryLinear for base-3 packed layers.

    Call this after load_model(..., strict=False) returns. base3_layer_names is
    the list stored in config.json under "ternary_base3_layers" at export time.

    Uses MLX's update_modules / tree_unflatten pattern (same as bitnet_quantize).
    """
    from mlx.utils import tree_flatten, tree_unflatten

    layer_set = set(base3_layer_names)
    replacements = []

    for name, module in tree_flatten(
        model.leaf_modules(), is_leaf=mnn.Module.is_module
    ):
        if name in layer_set:
            weight = module["weight"]   # 1-D uint32 (base-3 packed)
            scales = module["scales"]   # float16 [out, in//128]
            replacements.append((name, TernaryLinear(weight, scales)))

    if replacements:
        model.update_modules(tree_unflatten(replacements))
    return model


class TernaryLinear(mnn.Module):
    """Linear layer backed by base-3 packed ternary weights.

    Weights are stored as uint32 (packed) and decoded on the fly on the GPU.
    Scales are float16, one per group of 128 weights.

    Attributes:
        weight: uint32 array, shape [num_words] — base-3 packed codes.
        scales: float16 array, shape [out_features, in_features // 128].
    """

    def __init__(self, weight: mx.array, scales: mx.array) -> None:
        super().__init__()
        self.weight = weight
        self.scales = scales

    def __call__(self, x: mx.array) -> mx.array:
        w = dequantize_base3(self.weight, self.scales)
        return x @ w.T
