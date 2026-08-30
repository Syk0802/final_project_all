#!/usr/bin/env bash
# Stack baseline (top) and oval-erasing (bottom) ranklist images.
# Usage:
#   bash concat_ranklist.sh
# Adjust the paths / gap below as needed.

TOP_DIR="/xxx/extrawork/code/svc_net/ranklist/market/baseline"
BOT_DIR="/xxx/extrawork/code/svc_net/ranklist/market/market_01_oval_single"
OUT_DIR="/xxx/extrawork/code/svc_net/ranklist/market/compare_baseline_vs_oval"
GAP_PX=12
GAP_COLOR="white"

python3 concat_ranklist.py \
    --top "${TOP_DIR}" \
    --bottom "${BOT_DIR}" \
    --output "${OUT_DIR}" \
    --gap "${GAP_PX}" \
    --gap-color "${GAP_COLOR}"
