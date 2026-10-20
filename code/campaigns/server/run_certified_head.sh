#!/usr/bin/env bash
# Launch the certified-head comparison on the server in a detached tmux session.
# Three workers x 4 threads; each writes its own folder (merged by the analysis).
# Safe to re-run: finished cases are skipped. usage (on server): bash campaigns/server/run_certified_head.sh <out_dir>
set -euo pipefail
cd ~/commitjl
OUT=${1:-results/certified_head_20261017}
mkdir -p "$OUT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
groups=("gisette mnist" "fashion_mnist gaussian_hard" "20ng_tfidf_nonzero 20ng_tfidf_nonzero_k20")
tmux kill-session -t chead 2>/dev/null || true
for i in 0 1 2; do
  cmd=".venv/bin/python -m experiments.certified_head_comparison --out $OUT/worker$i --only ${groups[$i]} > $OUT/worker$i.log 2>&1; echo EXIT \$? >> $OUT/worker$i.log"
  run="OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 $cmd"
  # The first worker creates the session; a session with an exited first window would vanish.
  if [[ $i == 0 ]]; then tmux new-session -d -s chead -n w0 "$run"; else tmux new-window -t chead -n "w$i" "$run"; fi
done
sleep 2; tmux list-windows -t chead
