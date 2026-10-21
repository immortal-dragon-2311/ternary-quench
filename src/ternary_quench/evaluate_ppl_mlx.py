#!/usr/bin/env python3
"""Measure WikiText-2 word perplexity on an MLX model via lm_eval."""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from importlib.metadata import version
from pathlib import Path

import numpy as np

# lm_eval 0.4.7 imports hf_vlms which references transformers.AutoModelForVision2Seq,
# removed in transformers 5. Stub the module before any lm_eval import.
sys.modules.setdefault(
    "lm_eval.models.hf_vlms", types.ModuleType("lm_eval.models.hf_vlms")
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    np.random.seed(args.seed)

    import lm_eval
    import mlx.core as mx
    from lm_eval.tasks import TaskManager
    from mlx_lm.evaluate import MLXLM

    mx.random.seed(args.seed)
    model = MLXLM(
        str(args.model),
        batch_size=args.batch_size,
        use_chat_template=False,
    )
    evaluated = lm_eval.simple_evaluate(
        model=model,
        tasks=["wikitext"],
        batch_size=args.batch_size,
        num_fewshot=0,
        task_manager=TaskManager(include_defaults=True),
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
    )
    raw = evaluated["results"]["wikitext"]
    word_ppl = float(raw["word_perplexity,none"])
    byte_ppl = float(raw.get("byte_perplexity,none", float("nan")))

    metrics = {
        "word_perplexity": round(word_ppl, 4),
        "byte_perplexity": round(byte_ppl, 4),
    }
    payload = {
        "label": args.label,
        "model": str(args.model.resolve()),
        "protocol": {
            "task": "wikitext",
            "num_fewshot": 0,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "lm_eval": version("lm-eval"),
            "mlx_lm": version("mlx-lm"),
        },
        "metrics": metrics,
        "raw": raw,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    print(args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
