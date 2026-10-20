#!/usr/bin/env bash
# Fixed-centre audit study on the server, in a detached tmux session (resumable).
# usage (on server): bash campaigns/server/run_protected_fixed_center.sh <out_dir>
set -euo pipefail
cd ~/commitjl
OUT=${1:-results/protected_fixed_center_20261017}
mkdir -p "$OUT"
groups=("gisette arcene" "mnist fashion_mnist" "20ng_tfidf_nonzero gaussian_easy gaussian_hard dbscan_margin")
tmux kill-session -t pfc 2>/dev/null || true
for i in 0 1 2; do
  run="OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 .venv/bin/python -m experiments.protected_fixed_center --out $OUT/worker$i --only ${groups[$i]} > $OUT/worker$i.log 2>&1; echo EXIT \$? >> $OUT/worker$i.log"
  if [[ $i == 0 ]]; then tmux new-session -d -s pfc -n w0 "$run"; else tmux new-window -t pfc -n "w$i" "$run"; fi
done
sleep 2; tmux list-windows -t pfc
