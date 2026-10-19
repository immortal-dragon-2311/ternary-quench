"""
Monitor CPU usage + disk IO during training.
Run in a separate terminal alongside monitor.py.

Usage:
    python3 scripts/monitor_cpu.py
    python3 scripts/monitor_cpu.py --interval 5
"""

import argparse
import signal
import sys
import time

import psutil

trace = []
start_time = time.perf_counter()


def sample():
    t       = round(time.perf_counter() - start_time, 2)
    cpu_pct = psutil.cpu_percent(interval=None)
    disk    = psutil.disk_io_counters()
    return t, cpu_pct, disk


def handle_exit(sig, frame):
    print("\nStopping.")
    try:
        import matplotlib.pyplot as plt
        ts   = [r[0] / 60 for r in trace]
        cpus = [r[1] for r in trace]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(ts, cpus, color="steelblue", linewidth=1.5, label="CPU %")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("CPU usage (%)")
        ax.set_title("CPU usage during training — Qwen3-1.7B MPS")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("cpu_trace.png", dpi=150)
        print("Plot saved to cpu_trace.png")
        plt.show()
    except ImportError:
        pass
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()

    signal.signal(signal.SIGINT,  handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # Prime cpu_percent (first call always returns 0)
    psutil.cpu_percent(interval=None)
    prev_disk = psutil.disk_io_counters()

    print(f"Monitoring CPU + disk IO every {args.interval}s — Ctrl+C to stop\n")
    print(f"{'Time':>8}  {'CPU %':>7}  {'Disk read MB/s':>15}  {'Disk write MB/s':>16}")
    print("-" * 55)

    while True:
        time.sleep(args.interval)
        t, cpu_pct, disk = sample()
        trace.append((t, cpu_pct))

        read_mb  = (disk.read_bytes  - prev_disk.read_bytes)  / 1024**2 / args.interval
        write_mb = (disk.write_bytes - prev_disk.write_bytes) / 1024**2 / args.interval
        prev_disk = disk

        print(f"{t:>8.1f}s  {cpu_pct:>6.1f}%  {read_mb:>15.2f}  {write_mb:>16.2f}")
