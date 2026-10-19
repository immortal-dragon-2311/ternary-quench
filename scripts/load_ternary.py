"""
Load a ternary-quench .pt checkpoint into a HuggingFace model.

The ternary.pt file is a dict:
  key:   layer name e.g. "model.layers.0.self_attn.q_proj"
  value: dict with keys:
    codes  (int8,    shape [num_groups, group_size])
    scales (float32, shape [num_groups, 1])
    + training artifacts (lora_A, lora_B, t_scale, t_mu, t_round) — ignored at inference

Reconstruction per group of 128 weights:
    weight = (codes * scales).reshape(out_features, in_features)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def reconstruct_weight(
    codes: torch.Tensor,
    scales: torch.Tensor,
    target_shape: tuple[int, int],
) -> torch.Tensor:
    """
    Reconstruct a float weight matrix from ternary codes + per-group scales.

    codes:  int8,    shape [num_groups, group_size]  values in {-1, 0, +1}
    scales: float32, shape [num_groups, 1]
    target_shape: (out_features, in_features)
    """
    out_f, in_f = target_shape
    total = out_f * in_f

    if codes.numel() != total:
        raise ValueError(
            f"codes has {codes.numel()} elements, "
            f"expected {total} for shape {target_shape}"
        )

    group_size = codes.shape[-1] if codes.dim() > 1 else total
    num_groups = total // group_size

    grouped = codes.float().reshape(num_groups, group_size)  # [G, 128]
    reconstructed = (grouped * scales).reshape(out_f, in_f)  # [out, in]
    return reconstructed


def load_ternary_weights(
    model: nn.Module,
    ternary_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, str]:
    """
    Apply ternary weights from a ternary.pt checkpoint into a model in-place.

    Returns a report dict: {layer_name -> "replaced" | reason for skipping}
    Raises RuntimeError if any layer was unexpectedly skipped.
    """
    print(f"Loading checkpoint from {ternary_path} ...")
    ternary_state = torch.load(ternary_path, map_location="cpu", weights_only=False)
    print(f"  {len(ternary_state)} entries in checkpoint")

    report: dict[str, str] = {}

    with torch.no_grad():
        for layer_name, record in ternary_state.items():
            # keys in ternary.pt: "model.layers.0.self_attn.q_proj"
            # parameter in model: "model.layers.0.self_attn.q_proj.weight"
            param_path = layer_name + ".weight"

            # Navigate the module tree
            parts = param_path.split(".")
            try:
                module = model
                for part in parts[:-1]:
                    module = getattr(module, part)
                param = getattr(module, parts[-1])
            except AttributeError:
                report[layer_name] = f"NOT FOUND in model ({param_path})"
                continue

            if not isinstance(param, torch.Tensor):
                report[layer_name] = f"NOT A TENSOR ({type(param)})"
                continue

            codes  = record["codes"]   # int8
            scales = record["scales"]  # float32

            try:
                reconstructed = reconstruct_weight(codes, scales, tuple(param.shape))
            except ValueError as e:
                report[layer_name] = f"SHAPE MISMATCH: {e}"
                continue

            param.copy_(reconstructed.to(dtype=dtype, device=param.device))
            report[layer_name] = "replaced"

    replaced = sum(1 for v in report.values() if v == "replaced")
    problems = {k: v for k, v in report.items() if v != "replaced"}

    print(f"  {replaced}/{len(report)} layers replaced")

    if problems:
        for name, reason in problems.items():
            print(f"  PROBLEM: {name} — {reason}")
        raise RuntimeError(
            f"{len(problems)} layer(s) were not replaced — "
            "model ID may not match checkpoint, or checkpoint is corrupt."
        )

    return report
