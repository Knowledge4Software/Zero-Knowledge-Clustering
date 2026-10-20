cd ~/commitjl
R=results/paper_runs_20261019
while tmux has-session -t smoke 2>/dev/null; do sleep 30; done
cp -r /tmp/rw_smoke $R/smoke_mnist_n8 2>/dev/null
B=zk/bin/commitjl-bound-reuse-linux
export GOMAXPROCS=4 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
step() { echo "$(date -Is) START $1" >> $R/queue.log; shift; "$@"; echo "$(date -Is) END rc=$?" >> $R/queue.log; }
step "A+B+E mnist n64" bash -c ".venv/bin/python -m experiments.reuse_workload --bin $B --out $R/reuse_mnist_n64 --dataset mnist --n 64 --b 64 --reps 3 > $R/reuse_mnist_n64.log 2>&1"
step "C cifar n64" bash -c ".venv/bin/python -m experiments.reuse_workload --bin $B --out $R/reuse_cifar_n64 --dataset cifar10_resnet50 --n 64 --b 64 --reps 3 > $R/reuse_cifar_n64.log 2>&1"
step "D mnist n256" bash -c ".venv/bin/python -m experiments.reuse_workload --bin $B --out $R/reuse_mnist_n256 --dataset mnist --n 256 --b 64 --reps 1 > $R/reuse_mnist_n256.log 2>&1"
groups=("gisette arcene" "mnist fashion_mnist" "20ng_tfidf_nonzero gaussian_easy gaussian_hard dbscan_margin")
step "F fidelity 5 starts" bash -c "for i in 0 1 2; do OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.protected_fixed_center --starts 5 --out $R/fidelity_5starts/worker\$i --only \$(echo '${groups[0]}|${groups[1]}|${groups[2]}' | cut -d'|' -f\$((i+1))) > $R/fidelity_w\$i.log 2>&1 & done; wait"
echo "$(date -Is) ALL_DONE" >> $R/queue.log
