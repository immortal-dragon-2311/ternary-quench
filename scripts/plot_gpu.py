"""
Plot GPU metrics from gpu_trace.csv produced by log_gpu.sh.

Usage:
    python3 scripts/plot_gpu.py
    python3 scripts/plot_gpu.py --csv gpu_trace.csv --out gpu_plot.png
"""

import argparse
import csv

import matplotlib.pyplot as plt


def load(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "t":     float(row["elapsed_sec"]) / 60,   # minutes
                "pct":   float(row["gpu_active_pct"]),
                "freq":  float(row["gpu_freq_mhz"]),
                "power": float(row["gpu_power_mw"]),
            })
    return rows


def plot(rows, out_path):
    ts    = [r["t"]     for r in rows]
    pcts  = [r["pct"]   for r in rows]
    freqs = [r["freq"]  for r in rows]
    pwrs  = [r["power"] for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(ts, pcts,  color="steelblue", linewidth=1.5)
    axes[0].set_ylabel("GPU active (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ts, freqs, color="darkorange", linewidth=1.5)
    axes[1].set_ylabel("GPU freq (MHz)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ts, pwrs,  color="tomato", linewidth=1.5)
    axes[2].set_ylabel("GPU power (mW)")
    axes[2].set_xlabel("Time (minutes)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("GPU metrics during training — Qwen3-1.7B MPS", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="gpu_trace.csv")
    parser.add_argument("--out", default="gpu_plot.png")
    args = parser.parse_args()

    rows = load(args.csv)
    print(f"Loaded {len(rows)} samples from {args.csv}")
    plot(rows, args.out)
