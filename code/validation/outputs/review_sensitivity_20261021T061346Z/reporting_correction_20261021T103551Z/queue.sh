#!/usr/bin/env bash
set -u
R=/root/commitjl/results/review_sensitivity_20261021T061346Z
F=/root/commitjl/results/review_sensitivity_20261021T061346Z/reporting_correction_20261021T103551Z
exec 9>/root/commitjl/results/review-proof-campaign.lock
echo "$(date -Is) WAIT_FOR_PROOF_CAMPAIGN" >> "$F/queue.log"
flock 9
export PYTHONPATH="$R/source"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/root/commitjl/.venv/bin/python "$F/review_sensitivity.py" finalize --prepared "$R/prepared" --runs "$R" --out "$R/completed_summary_corrected.json" > "$F/finalize.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ]; then echo "$(date -Is) ALL_DONE corrected_report" >> "$F/queue.log"; else echo "$(date -Is) FAILED_REPORT rc=$rc" >> "$F/queue.log"; fi
