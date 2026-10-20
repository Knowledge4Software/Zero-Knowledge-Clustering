#!/usr/bin/env bash
# New-centre reuse study (native, resumable) in a detached tmux session.
# 1) builds the neural-embedding corpora in a separate venv (.venv-embed), once;
# 2) runs four single-threaded workers over all datasets.
# usage (on server): bash campaigns/server/run_new_center_reuse.sh <out_dir>
set -euo pipefail
cd ~/commitjl
OUT=${1:-results/new_center_reuse_$(date +%Y%m%d)}
mkdir -p "$OUT"
cat > "$OUT/run.sh" <<EOF
set -uo pipefail
cd ~/commitjl
if [[ ! -x .venv-embed/bin/python ]]; then
  python3 -m venv .venv-embed
  .venv-embed/bin/pip install -q --upgrade pip
  .venv-embed/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
  .venv-embed/bin/pip install -q sentence-transformers scikit-learn
fi
OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 .venv-embed/bin/python -m data.build_embeddings cifar10_resnet50 20ng_minilm > $OUT/embeddings.log 2>&1
echo EMBED_EXIT \$? >> $OUT/embeddings.log
for i in 0 1 2 3; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    .venv/bin/python -m experiments.new_center_reuse --out $OUT --worker \$i --workers 4 > $OUT/worker\$i.log 2>&1 &
done
wait
echo RUN_DONE >> $OUT/embeddings.log
EOF
tmux kill-session -t ncr 2>/dev/null || true
tmux new-session -d -s ncr "bash $OUT/run.sh"
sleep 1; tmux ls
