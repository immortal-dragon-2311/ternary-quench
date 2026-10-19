"""
Benchmark a Qwen3-1.7B model: perplexity, disk size, continuous GPU memory trace.

Usage:
    # Float baseline
    uv run python benchmark.py --model Qwen/Qwen3-1.7B

    # After ternary quantization (uses released checkpoint)
    uv run python benchmark.py --model Qwen/Qwen3-1.7B \
        --ternary-checkpoint penkia/catq-qwen3-1.7b-ternary-st-epoch60-boundary-depth28

    # With a local ternary.pt
    uv run python benchmark.py --model Qwen/Qwen3-1.7B \
        --ternary-checkpoint path/to/ternary.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time

import psutil
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Memory sampler ────────────────────────────────────────────────────────────

class MemorySampler:
    """
    Samples both GPU (torch.cuda.memory_allocated) and CPU process RSS in a
    background thread.

    Each sample is a dict:
        {"t": elapsed_sec, "gpu_gb": float, "cpu_gb": float}

    cpu_gb = RSS of this process (torch tensors on CPU + Python heap + model
             weights that were offloaded).  It is the number you'd see in
             `htop` for this PID.

    Usage:
        with MemorySampler(interval_ms=50) as s:
            do_work()
        print(s.gpu_peak_gb, s.cpu_peak_gb, s.trace)
    """

    def __init__(self, interval_ms: int = 50):
        self.interval = interval_ms / 1000.0
        self.trace: list[dict] = []
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        t0 = time.perf_counter()
        while not self._stop.is_set():
            gpu_gb = (torch.cuda.memory_allocated() / 1024**3
                      if torch.cuda.is_available() else 0.0)
            cpu_gb = self._proc.memory_info().rss / 1024**3
            self.trace.append({
                "t":      round(time.perf_counter() - t0, 4),
                "gpu_gb": round(gpu_gb, 4),
                "cpu_gb": round(cpu_gb, 4),
            })
            time.sleep(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def _peak(self, key: str) -> float:
        return max((s[key] for s in self.trace), default=0.0)

    def _mean(self, key: str) -> float:
        if not self.trace:
            return 0.0
        return sum(s[key] for s in self.trace) / len(self.trace)

    @property
    def gpu_peak_gb(self) -> float:
        return self._peak("gpu_gb")

    @property
    def gpu_mean_gb(self) -> float:
        return self._mean("gpu_gb")

    @property
    def cpu_peak_gb(self) -> float:
        return self._peak("cpu_gb")

    @property
    def cpu_mean_gb(self) -> float:
        return self._mean("cpu_gb")


# ── Metrics ───────────────────────────────────────────────────────────────────

def measure_perplexity(
    model: torch.nn.Module,
    tokenizer,
    device: str,
    n_tokens: int = 4096,
    ctx: int = 512,
) -> float:
    """Perplexity on WikiText-2 test set (lower = better)."""
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = torch.tensor(ids[:n_tokens], dtype=torch.long)

    n_chunks = ids.numel() // ctx
    total_nll = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for i in range(n_chunks):
            chunk = ids[i * ctx : (i + 1) * ctx].unsqueeze(0).to(device)
            out = model(chunk[:, :-1], use_cache=False)
            logits = out.logits.float()
            nll = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                chunk[:, 1:].reshape(-1),
                reduction="sum",
            )
            total_nll += float(nll)
            total_tokens += chunk.shape[1] - 1

    return math.exp(total_nll / total_tokens)


def measure_memory_during_generate(
    model: torch.nn.Module,
    tokenizer,
    device: str,
    interval_ms: int = 50,
) -> MemorySampler:
    """
    Run a single generate() pass and record GPU memory every `interval_ms` ms.
    Returns the MemorySampler with .trace, .peak_gb, .mean_gb populated.
    """
    if not torch.cuda.is_available():
        return MemorySampler()

    prompt = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with MemorySampler(interval_ms=interval_ms) as sampler:
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=50)

    return sampler


def model_disk_size_gb(model_id: str) -> float:
    """Total size of all cached model files (safetensors + config etc.)."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(model_id)
    total = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(path)
        for f in files
    )
    return total / 1024**3


def ternary_checkpoint_size_gb(checkpoint: str) -> float:
    if os.path.isfile(checkpoint):
        return os.path.getsize(checkpoint) / 1024**3
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(repo_id=checkpoint, filename="ternary.pt")
    return os.path.getsize(local) / 1024**3


def resolve_ternary_path(checkpoint: str) -> str:
    if os.path.isfile(checkpoint):
        return checkpoint
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=checkpoint, filename="ternary.pt")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-1.7B float vs ternary")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--ternary-checkpoint",
        default=None,
        help="HF repo ID or local path to ternary.pt. Omit to benchmark float only.",
    )
    parser.add_argument("--n-tokens", type=int, default=4096)
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--memory-interval-ms", type=int, default=50,
                        help="Memory sampling interval in ms (default 50)")
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results: dict = {}

    # ── Float baseline ─────────────────────────────────────────────────────────
    print(f"\n[Float] Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device
    )

    print("[Float] Measuring disk size ...")
    float_disk = model_disk_size_gb(args.model)
    print(f"  disk size:   {float_disk:.2f} GB")

    print("[Float] Sampling memory during inference ...")
    float_sampler = measure_memory_during_generate(
        model, tokenizer, device, args.memory_interval_ms
    )
    print(f"  gpu peak:    {float_sampler.gpu_peak_gb:.2f} GB  "
          f"mean: {float_sampler.gpu_mean_gb:.2f} GB")
    print(f"  cpu peak:    {float_sampler.cpu_peak_gb:.2f} GB  "
          f"mean: {float_sampler.cpu_mean_gb:.2f} GB  "
          f"samples: {len(float_sampler.trace)}")

    print("[Float] Measuring perplexity ...")
    float_ppl = measure_perplexity(model, tokenizer, device, args.n_tokens, args.ctx)
    print(f"  perplexity:  {float_ppl:.2f}")

    results["float"] = {
        "model": args.model,
        "dtype": "bfloat16",
        "disk_gb": round(float_disk, 3),
        "gpu_peak_gb": round(float_sampler.gpu_peak_gb, 3),
        "gpu_mean_gb": round(float_sampler.gpu_mean_gb, 3),
        "cpu_peak_gb": round(float_sampler.cpu_peak_gb, 3),
        "cpu_mean_gb": round(float_sampler.cpu_mean_gb, 3),
        "memory_trace": float_sampler.trace,   # [{"t", "gpu_gb", "cpu_gb"}, ...]
        "perplexity": round(float_ppl, 4),
    }

    del model
    torch.cuda.empty_cache()

    # ── Ternary ────────────────────────────────────────────────────────────────
    if args.ternary_checkpoint:
        from load_ternary import load_ternary_weights

        print(f"\n[Ternary] Checkpoint: {args.ternary_checkpoint}")

        print("[Ternary] Measuring checkpoint disk size ...")
        ternary_disk = ternary_checkpoint_size_gb(args.ternary_checkpoint)
        print(f"  disk size:   {ternary_disk:.2f} GB")

        print("[Ternary] Loading base model ...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map=device
        )

        print("[Ternary] Applying ternary weights ...")
        ternary_path = resolve_ternary_path(args.ternary_checkpoint)
        load_ternary_weights(model, ternary_path, device=device, dtype=torch.bfloat16)

        print("[Ternary] Sampling memory during inference ...")
        ternary_sampler = measure_memory_during_generate(
            model, tokenizer, device, args.memory_interval_ms
        )
        print(f"  gpu peak:    {ternary_sampler.gpu_peak_gb:.2f} GB  "
              f"mean: {ternary_sampler.gpu_mean_gb:.2f} GB")
        print(f"  cpu peak:    {ternary_sampler.cpu_peak_gb:.2f} GB  "
              f"mean: {ternary_sampler.cpu_mean_gb:.2f} GB  "
              f"samples: {len(ternary_sampler.trace)}")

        print("[Ternary] Measuring perplexity ...")
        ternary_ppl = measure_perplexity(model, tokenizer, device, args.n_tokens, args.ctx)
        print(f"  perplexity:  {ternary_ppl:.2f}")

        results["ternary"] = {
            "checkpoint": args.ternary_checkpoint,
            "disk_gb": round(ternary_disk, 3),
            "gpu_peak_gb": round(ternary_sampler.gpu_peak_gb, 3),
            "gpu_mean_gb": round(ternary_sampler.gpu_mean_gb, 3),
            "cpu_peak_gb": round(ternary_sampler.cpu_peak_gb, 3),
            "cpu_mean_gb": round(ternary_sampler.cpu_mean_gb, 3),
            "memory_trace": ternary_sampler.trace,
            "perplexity": round(ternary_ppl, 4),
        }

        del model
        torch.cuda.empty_cache()

    # ── Summary table ──────────────────────────────────────────────────────────
    has_ternary = "ternary" in results
    print("\n" + "=" * 60)
    header = f"{'Metric':<28} {'Float BF16':>14}"
    if has_ternary:
        header += f" {'Ternary':>14}"
    print(header)
    print("=" * 60)

    fr = results["float"]
    tr = results.get("ternary", {})

    def row(label: str, key: str, fmt: str = ".2f") -> None:
        line = f"{label:<28} {fr.get(key, float('nan')):>14{fmt}}"
        if has_ternary:
            line += f" {tr.get(key, float('nan')):>14{fmt}}"
        print(line)

    row("Disk size (GB)",        "disk_gb")
    row("GPU peak (GB)",         "gpu_peak_gb")
    row("GPU mean (GB)",         "gpu_mean_gb")
    row("CPU peak (GB)",         "cpu_peak_gb")
    row("CPU mean (GB)",         "cpu_mean_gb")
    row("Perplexity (↓ better)", "perplexity", ".4f")
    print("=" * 60)

    if has_ternary:
        compression = fr["disk_gb"] / tr["disk_gb"]
        gpu_red = (fr["gpu_peak_gb"] / tr["gpu_peak_gb"]
                   if tr["gpu_peak_gb"] else float("nan"))
        cpu_red = (fr["cpu_peak_gb"] / tr["cpu_peak_gb"]
                   if tr["cpu_peak_gb"] else float("nan"))
        ppl_delta = tr["perplexity"] - fr["perplexity"]
        print(f"\nCompression ratio:        {compression:.2f}x")
        print(f"GPU memory reduction:     {gpu_red:.2f}x")
        print(f"CPU memory reduction:     {cpu_red:.2f}x")
        print(f"Perplexity degradation:   +{ppl_delta:.4f}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print("  memory_trace: list of {{t, gpu_gb, cpu_gb}} samples at "
          f"{args.memory_interval_ms}ms intervals")


if __name__ == "__main__":
    main()
