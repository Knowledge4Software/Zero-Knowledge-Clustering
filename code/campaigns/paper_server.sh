#!/usr/bin/env bash
# Ubuntu preset. Usage: bash campaigns/paper_server.sh rq1|pilot|native|selected|proof
set -euo pipefail
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_directory/.."
python_bin="${PYTHON:-python}"
mode="${1:-help}"
common=(--run --seeds 10 --starts 5 --jobs 4 --threads 4 --proof-threads 16
        --proof-rss-gib 40 --task-rss-gib 48 --bootstrap 2000)
case "$mode" in
  rq1)
    "$python_bin" -m experiments.rq1_complete --run --seeds 10 --starts 5 \
      --jobs 4 --threads 4 --max-rss-gib 8 \
      --out results/rq1/runs/paper_balanced_v3
    exec "$python_bin" -m analysis.rq1_export \
      --runs results/rq1/runs/paper_balanced_v3 ;;
  pilot)
    exec "$python_bin" -m experiments.rq2345_complete "${common[@]}" \
      --stages rq2_pilot --proof-reps 3 \
      --out results/complete_rq2345/proof_pilot_v1 ;;
  native)
    exec "$python_bin" -m experiments.rq2345_complete "${common[@]}" \
      --stages rq3_native rq3_grinding rq4_centroid rq4_dbscan rq4_phase rq5_native rq5_precision \
      --out results/complete_rq2345/native_balanced_v1 ;;
  selected)
    exec "$python_bin" -m experiments.rq2345_complete "${common[@]}" \
      --stages rq2_quality rq2_selected --proof-reps 10 \
      --out results/complete_rq2345/selected_balanced_v1 ;;
  proof)
    exec "$python_bin" -m experiments.rq2345_complete "${common[@]}" \
      --stages rq3_proof rq5_proof rq5_ablation --proof-reps 10 \
      --out results/complete_rq2345/proof_sensitivity_v1 ;;
  *)
    echo "Usage: bash campaigns/paper_server.sh rq1|pilot|native|selected|proof"
    echo "See docs/rq2345_complete.md for scope, subsets and resume commands."
    exit 2 ;;
esac
