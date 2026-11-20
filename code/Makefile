.PHONY: help zk golden test test-zk paper evidence all

PY ?= python
REPLAY_OUT ?= ../output/reviewer_replay

help:
	@echo "Targets: test, zk, test-zk, paper, evidence, golden"
	@echo "evidence requires a fresh REPLAY_OUT directory. No campaign runs by default."

# Build the legacy projection CLI and the current update-certificate CLI.
zk:
	cd zk && go build -o bin/commitjl-prove ./cmd/commitjl-prove
	cd zk && go build -o bin/commitjl-bound ./cmd/commitjl-bound

# Regenerate the MiMC golden vectors used by tests/test_hashing.py to
# cross-check Python's MiMC implementation against gnark-crypto's.
golden:
	cd zk && go run ./cmd/golden > ../tests/golden/mimc.json

test:
	$(PY) -m pytest tests -q

test-zk: zk
	$(PY) -m pytest tests -k zk -q

paper:
	$(PY) -m analysis.build_current_paper

evidence: zk
	$(PY) ../reviewer_artifact/check.py --binary zk/bin/commitjl-bound --out $(REPLAY_OUT)

all: test paper
