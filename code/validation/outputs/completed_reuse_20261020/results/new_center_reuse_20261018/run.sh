set -uo pipefail
cd ~/commitjl
if [[ ! -x .venv-embed/bin/python ]]; then
  python3 -m venv .venv-embed
  .venv-embed/bin/pip install -q --upgrade pip
  .venv-embed/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
  .venv-embed/bin/pip install -q sentence-transformers scikit-learn
fi
OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 .venv-embed/bin/python -m data.build_embeddings cifar10_resnet50 20ng_minilm > results/new_center_reuse_20261018/embeddings.log 2>&1
echo EMBED_EXIT $? >> results/new_center_reuse_20261018/embeddings.log
for i in 0 1 2 3; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1     .venv/bin/python -m experiments.new_center_reuse --out results/new_center_reuse_20261018 --worker $i --workers 4 > results/new_center_reuse_20261018/worker$i.log 2>&1 &
done
wait
echo RUN_DONE >> results/new_center_reuse_20261018/embeddings.log
