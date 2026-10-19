#!/usr/bin/env bash
# Log GPU metrics to gpu_trace.csv using powermetrics.
# Must be run with sudo.
#
# Usage:
#   sudo bash scripts/log_gpu.sh
#   sudo bash scripts/log_gpu.sh --interval 10000 --out my_trace.csv

INTERVAL=10000   # ms
OUT="gpu_trace.csv"

while [[ $# -gt 0 ]]; do
  case $1 in
    --interval) INTERVAL="$2"; shift 2 ;;
    --out)      OUT="$2";      shift 2 ;;
    *) shift ;;
  esac
done

echo "timestamp,elapsed_sec,gpu_active_pct,gpu_freq_mhz,gpu_power_mw" > "$OUT"
echo "Logging GPU metrics every $((INTERVAL/1000))s to $OUT — Ctrl+C to stop"
echo ""
printf "%-10s %-14s %-14s %-14s\n" "Time" "GPU active %" "GPU freq MHz" "GPU power mW"
echo "------------------------------------------------------"

START=$(date +%s)
gpu_pct=""
gpu_freq=""

sudo powermetrics --samplers gpu_power -i "$INTERVAL" 2>/dev/null | \
while IFS= read -r line; do
  if [[ "$line" == *"GPU HW active residency"* ]]; then
    gpu_pct=$(echo "$line" | grep -oE '[0-9]+\.[0-9]+' | head -1)

  elif [[ "$line" == *"GPU HW active frequency"* ]]; then
    gpu_freq=$(echo "$line" | grep -oE '[0-9]+' | head -1)

  elif [[ "$line" == *"GPU Power"* ]]; then
    gpu_power=$(echo "$line" | grep -oE '[0-9]+' | head -1)
    ts=$(date +%s)
    elapsed=$(( ts - START ))
    time_str=$(date +%H:%M:%S)

    echo "$ts,$elapsed,${gpu_pct:-0},${gpu_freq:-0},${gpu_power:-0}" >> "$OUT"
    printf "%-10s %-14s %-14s %-14s\n" \
      "$time_str" "${gpu_pct:-?}%" "${gpu_freq:-?}" "${gpu_power:-?}"
  fi
done
