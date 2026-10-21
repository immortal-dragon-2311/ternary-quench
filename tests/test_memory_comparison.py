"""Compare peak Metal memory and inference runtime across quantization formats."""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm
import pytest

PROMPT = "The capital of France is"
MAX_TOKENS = 20

MODELS = {
    "mlx-4bit":          Path("out/qwen3-1.7b/mlx-4bit"),
    "mlx-ternary-2bit":  Path("out/qwen3-1.7b/mlx-ternary"),
    "mlx-ternary-base3": Path("out/qwen3-1.7b/mlx-ternary-base3"),
}


def _load(model_dir: Path):
    from mlx_lm.utils import load_model, load_tokenizer
    model, config = load_model(model_dir, strict=False)
    tokenizer = load_tokenizer(model_dir)
    if "ternary_base3_layers" in config:
        from ternary_quench.decode_base3 import patch_base3
        model = patch_base3(model, config["ternary_base3_layers"])
    return model, tokenizer


def _measure(model, tokenizer) -> dict:
    # warm-up: one token to trigger Metal kernel compilation
    _ = mlx_lm.generate(model, tokenizer, prompt=PROMPT, max_tokens=1, verbose=False)
    mx.eval()

    mx.reset_peak_memory()
    t0 = time.perf_counter()
    response = mlx_lm.generate(
        model, tokenizer,
        prompt=PROMPT,
        max_tokens=MAX_TOKENS,
        verbose=False,
    )
    mx.eval()
    elapsed = time.perf_counter() - t0

    peak_gb = mx.get_peak_memory() / 1e9
    tokens_per_sec = MAX_TOKENS / elapsed
    return {
        "response": response,
        "peak_gb": peak_gb,
        "elapsed_s": elapsed,
        "tok_per_sec": tokens_per_sec,
    }


@pytest.mark.parametrize("name,model_dir", list(MODELS.items()))
def test_inference_benchmark(name, model_dir):
    model, tokenizer = _load(model_dir)
    m = _measure(model, tokenizer)
    print(
        f"\n{name}:"
        f"  peak={m['peak_gb']:.3f} GB"
        f"  speed={m['tok_per_sec']:.1f} tok/s"
        f"  elapsed={m['elapsed_s']:.2f}s"
        f"  output={m['response']!r}"
    )
    assert m["peak_gb"] > 0
    assert m["tok_per_sec"] > 0
    assert len(m["response"]) > 0
