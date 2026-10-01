.PHONY: zk golden test test-zk smoke native core tables figures analysis audit layout all

PY ?= python
export COMMITJL_RESULTS_DIR

# Build the gnark-based ZK circuit CLI (code/zk/cmd/commitjl-prove).
zk:
	cd zk && go build -o bin/commitjl-prove ./cmd/commitjl-prove

# Regenerate the MiMC golden vectors used by tests/test_hashing.py to
# cross-check Python's MiMC implementation against gnark-crypto's.
golden:
	cd zk && go run ./cmd/golden > ../tests/golden/mimc.json

test:
	$(PY) -m pytest tests -q

test-zk: zk
	$(PY) -m pytest tests -k zk -q

# --- Experiment grids (plan P4) -------------------------------------------

smoke: COMMITJL_RESULTS_DIR = results/smoke
smoke: zk
	$(PY) -m experiments.rq1_fidelity --config experiments/configs/smoke.yaml
	$(PY) -m experiments.rq2_proof_cost --config experiments/configs/smoke.yaml
	$(PY) -m experiments.rq3_overhead --config experiments/configs/smoke.yaml
	$(PY) -m experiments.rq3_grinding --config experiments/configs/smoke.yaml --n-trials 8 --Q-list 1,2,4,8
	$(PY) -m experiments.rq4_certificates --config experiments/configs/smoke.yaml
	$(PY) -m experiments.rq5_scaling --config experiments/configs/smoke.yaml
	$(PY) -m experiments.ablation --config experiments/configs/smoke.yaml
	$(PY) -m experiments.summarize

native: COMMITJL_RESULTS_DIR = results/native
native:
	$(PY) -m experiments.rq1_fidelity --config experiments/configs/native.yaml
	$(PY) -m experiments.rq3_overhead --config experiments/configs/native.yaml
	$(PY) -m experiments.rq4_certificates --config experiments/configs/native.yaml --phase-diagram
	$(PY) -m experiments.rq5_scaling --config experiments/configs/native.yaml

core: COMMITJL_RESULTS_DIR = results/core
core: zk
	$(PY) -m experiments.rq2_proof_cost --config experiments/configs/core.yaml
	$(PY) -m experiments.ablation --config experiments/configs/core.yaml

# --- Tables & figures (plan P5) --------------------------------------------

tables:
	$(PY) -m experiments.make_tables --table all

figures:
	$(PY) -m experiments.make_figures --fig all
	$(PY) -m figures.schematics --diagram all

analysis:
	$(PY) -m analysis.paper_results
	$(PY) -m analysis.author_overview

audit:
	$(PY) validation/audit.py

layout:
	$(PY) validation/check_layout.py

all: smoke native core tables figures
