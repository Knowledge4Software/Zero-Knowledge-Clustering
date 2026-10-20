#!/usr/bin/env bash
# RQ2 per-query proof-cost benchmark on the server (one worker; resumable).
# usage (on server): bash campaigns/server/run_rq2_query.sh <out_dir>
set -euo pipefail
cd ~/commitjl
OUT=${1:-results/rq2_query_cost_20261017}
mkdir -p "$OUT"; chmod +x zk/bin/commitjl-query-linux
tmux kill-session -t rq2 2>/dev/null || true
run="GOGC=25 GOMAXPROCS=12 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 .venv/bin/python -m experiments.rq2_query_cost --bin zk/bin/commitjl-query-linux --out $OUT > $OUT/run.log 2>&1; echo EXIT \$? >> $OUT/run.log"
tmux new-session -d -s rq2 "$run"
sleep 2; tmux ls
