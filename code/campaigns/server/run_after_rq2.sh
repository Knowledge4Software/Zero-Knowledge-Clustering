#!/usr/bin/env bash
# Wait for the RQ2 timing run (tmux session rq2) to finish, then run the
# adversarial-attack study so it cannot disturb RQ2 proof timings.
# usage (on server): bash campaigns/server/run_after_rq2.sh <out_dir>
set -euo pipefail
cd ~/commitjl
OUT=${1:-results/adversarial_attack_20261017}
mkdir -p "$OUT"
tmux kill-session -t attack 2>/dev/null || true
run="while tmux has-session -t rq2 2>/dev/null; do sleep 60; done; OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 .venv/bin/python -m experiments.adversarial_attack --out $OUT > $OUT/run.log 2>&1; echo EXIT \$? >> $OUT/run.log"
tmux new-session -d -s attack "$run"
sleep 1; tmux ls
