"""
Monitor RAM + swap usage during training. Run in a separate terminal.
Ctrl+C to stop and save the plot.

Usage:
    python3 scripts/monitor.py
    python3 scripts/monitor.py --interval 2 --out memory_trace.png
"""

import argparse
import signal
import sys
import time
from datetime import datetime

import psutil

trace = []  # [(elapsed_sec, ram_gb, swap_gb), ...]
start_time = time.perf_counter()


def sample():
    vm   = psutil.virtual_memory()
    swap = psutil.swap_memory()
    t    = round(time.perf_counter() - start_time, 2)
    ram_gb  = round(vm.used  / 1024**3, 3)
    swap_gb = round(swap.used / 1024**3, 3)
    trace.append((t, ram_gb, swap_gb))
    return t, ram_gb, swap_gb


def save_plot(out_path: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot. Raw data saved to trace.txt")
        with open("trace.txt", "w") as f:
            f.write("elapsed_sec,ram_gb,swap_gb\n")
            for row in trace:
                f.write(",".join(map(str, row)) + "\n")
        return

    total_gb = psutil.virtual_memory().total / 1024**3
    ts    = [r[0] / 60 for r in trace]   # minutes
    rams  = [r[1] for r in trace]
    swaps = [r[2] for r in trace]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts, rams,  label="RAM used (GB)",  color="steelblue", linewidth=1.5)
    ax.plot(ts, swaps, label="Swap used (GB)", color="tomato",    linewidth=1.5)
    ax.axhline(total_gb, color="gray", linestyle="--", linewidth=1, label=f"Total RAM ({total_gb:.0f} GB)")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Memory (GB)")
    ax.set_title("Memory during training — Qwen3-1.7B MPS")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.show()


def handle_exit(sig, frame):
    print("\nStopping monitor...")
    save_plot(args.out)
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=5.0, help="Sampling interval in seconds")
    parser.add_argument("--out", default="memory_trace.png", help="Output plot path")
    args = parser.parse_args()

    signal.signal(signal.SIGINT,  handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    total_gb = psutil.virtual_memory().total / 1024**3
    print(f"Monitoring memory every {args.interval}s — Ctrl+C to stop and save plot")
    print(f"Total RAM: {total_gb:.1f} GB\n")
    print(f"{'Time':>8}  {'RAM (GB)':>10}  {'Swap (GB)':>10}  {'RAM %':>7}")
    print("-" * 42)

    while True:
        t, ram, swap = sample()
        pct = ram / total_gb * 100
        print(f"{t:>8.1f}s  {ram:>10.2f}  {swap:>10.2f}  {pct:>6.1f}%")
        time.sleep(args.interval)
