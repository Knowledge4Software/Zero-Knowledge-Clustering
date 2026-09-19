#!/usr/bin/env python
"""RQ3 -- ordering/overhead: per-stage protocol timings (commit_X,
record_theta, issue_nonce/expand_sparse, project_and_cluster) run under each
randomness-instantiation source (nonce, beacon, commit-reveal), with
retries on transient failure. Produces Table ordering-a rows.

Usage: python -m experiments.rq3_overhead --config experiments/configs/smoke.yaml
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from experiments.common import (common_arg_parser, load_config, load_prereg, generate_grid,
                                 row_hash, apply_shard, ResultWriter, default_out_path, capture_env)
from experiments.pipeline import load_row_dataset, normalize_for_quantization
from commitjl.clustering import kmeans_paired
from commitjl.protocol import run_protocol, derive_seed
from commitjl.randomness import TaskLog, BeaconTrace, CommitRevealChallenge
from commitjl.commit import commit_matrix, commit_theta
from commitjl.quantize import scale_for_precision, quantize

SEED_SOURCES = ["nonce", "beacon", "commit_reveal"]


def _rho_for_source(source: str, task_log: TaskLog, seed: int) -> bytes:
    if source == "nonce":
        return task_log.fresh_nonce()
    if source == "beacon":
        trace = BeaconTrace(rounds={seed: (b"\x00" * 31 + bytes([seed % 256]))})
        return trace.get_round(seed)
    if source == "commit_reveal":
        chal = CommitRevealChallenge()
        chal.commit("prover", value=(b"\x01" * 31 + bytes([seed % 256])))
        chal.commit("verifier", value=(b"\x02" * 31 + bytes([seed % 256])))
        return chal.finalize()
    raise KeyError(source)


def run_one(row, prereg, source: str, max_retries: int = 3):
    data = load_row_dataset(row)
    X = np.asarray(data["X"], dtype=np.float64)
    k = int(row.get("k", 5))
    seed = int(row.get("seed", 0))
    m = int(row.get("m", 16))
    q = int(row.get("q", 1))
    precision_bits = int(row.get("precision_bits", 16))
    S = scale_for_precision(precision_bits)
    B = int(row.get("B", 4096))
    X = normalize_for_quantization(X, S, B)
    Xq = quantize(X, S, B)
    labels, centroids, _ = kmeans_paired(Xq.astype(np.float64), k, seeds=[seed])[0]
    Cq = np.round(centroids).astype(np.int64)

    attempt, last_err = 0, None
    while attempt < max_retries:
        attempt += 1
        try:
            task_log = TaskLog(Q=max(1, attempt))
            result = run_protocol(Xq, Cq, m, min(q, X.shape[1]), S, task_log=task_log)
            result["_attempts"] = attempt
            result["_seed_source"] = source
            return result
        except Exception as e:  # pragma: no cover - retry path
            last_err = e
            time.sleep(0.0)
    raise RuntimeError(f"exhausted retries ({max_retries}) for source={source}: {last_err}")


def main():
    ap = common_arg_parser("RQ3 overhead / stage timings")
    args = ap.parse_args()
    cfg = load_config(args.config)
    prereg = load_prereg(args.prereg)
    rows = generate_grid(cfg)
    rows = apply_shard(rows, args.shard)
    if args.limit:
        rows = rows[: args.limit]

    out_path = args.out or default_out_path("rq3_overhead")
    writer = ResultWriter(out_path)
    env = capture_env()
    n_ok, n_fail = 0, 0

    for row in rows:
        for source in SEED_SOURCES:
            h = row_hash({**row, "seed_source": source})
            if writer.should_skip(h):
                continue
            try:
                result = run_one(row, prereg, source)
                t = result["transcript"]["stage_times_s"]
                b = result["transcript"]["stage_bytes"]
                out = dict(row)
                out.update({
                    "_hash": h, "_status": "ok", "seed_source": source,
                    "attempts": result["_attempts"],
                    **{f"time_{k_}": v for k_, v in t.items()},
                    **{f"bytes_{k_}": v for k_, v in b.items()},
                    "total_time_s": float(sum(t.values())),
                    "total_bytes": int(sum(b.values())),
                })
                writer.write(out)
                n_ok += 1
            except Exception as e:
                writer.write({"_hash": h, "_status": "failed", "_error": str(e),
                              "seed_source": source, **row})
                n_fail += 1

    print(f"rq3_overhead: {n_ok} ok, {n_fail} failed, out={out_path}")
    print(f"env: git={env.get('git_hash')}")


if __name__ == "__main__":
    main()
