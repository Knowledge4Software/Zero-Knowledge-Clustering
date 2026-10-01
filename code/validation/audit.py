"""Read-only, reproducible validation checks for the CommitJL artifact."""
from __future__ import annotations

import hashlib
import argparse
import inspect
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
RESULTS = CODE / "results"
PROOF_CONFIG = RESULTS / "rq2/runs/representative/selected_rq1_shapes.yaml"
if not PROOF_CONFIG.exists():
    PROOF_CONFIG = CODE / "experiments/configs/core_light.yaml"
sys.path.insert(0, str(CODE))

from commitjl.certificates import (  # noqa: E402
    centroid_margin_certificate,
    cost_bounds,
    dbscan_ambiguity_band,
)
from commitjl.clustering import nearest_centroid_assign, psi_hat  # noqa: E402
from commitjl.metrics import distortion_summary  # noqa: E402
from commitjl.quantize import (  # noqa: E402
    BN254_FR,
    QuantizationParams,
    pairwise_distance_perturbation_bound,
    quantize,
    quantize_error_bound,
    squared_distance_error_bound,
)
from commitjl.sparse_gen import sparse_matrix_gen  # noqa: E402


checks: list[dict] = []

PAPER_RESULT_PATHS = {
    "rq1_fidelity.parquet": "rq1/runs/native/rq1_fidelity.parquet",
    "rq2_proof_cost.parquet": "rq2/runs/representative/rq2_proof_cost.parquet",
    "rq3_overhead.parquet": "rq3/runs/native/rq3_overhead.parquet",
    "rq3_grinding.parquet": "rq3/runs/native/rq3_grinding.parquet",
    "rq3_paired.parquet": "rq3/runs/paired/rq3_paired.parquet",
    "rq4_certificates.parquet": "rq4/runs/native/rq4_certificates.parquet",
    "rq4_real_certificates.parquet": "rq4/runs/real/rq4_real_certificates.parquet",
    "rq5_scaling.parquet": "rq5/runs/native/rq5_scaling.parquet",
    "rq5_gamma.parquet": "rq5/runs/gamma/rq5_gamma.parquet",
    "rq5_precision.parquet": "rq5/runs/precision_20ref/rq5_precision.parquet",
    "rq5_precision_24_reference.parquet": "rq5/runs/precision_24ref/rq5_precision_24_reference.parquet",
}


def result_path(filename: str) -> Path:
    """Resolve the current per-RQ layout or an explicitly supplied flat run."""
    if (RESULTS / "rq1/runs").is_dir() and filename in PAPER_RESULT_PATHS:
        return RESULTS / PAPER_RESULT_PATHS[filename]
    return RESULTS / filename


def raw_result_paths() -> list[Path]:
    if (RESULTS / "rq1/runs").is_dir():
        return [RESULTS / path for path in PAPER_RESULT_PATHS.values()]
    return sorted(RESULTS.glob("*.parquet"))


def record(name: str, status: str, evidence, detail: str = "") -> None:
    checks.append({"name": name, "status": status, "evidence": evidence, "detail": detail})


def sha_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def check_sparse_generator() -> None:
    failures = []
    for m in (1, 2, 7, 16, 31):
        for q in sorted({1, min(2, m), m}):
            for d in (1, 9):
                records, a = sparse_matrix_gen(b"audit-seed", m, d, q)
                if len(records) != d * q:
                    failures.append(f"records m={m},q={q},d={d}")
                if not np.all(np.count_nonzero(a, axis=0) == q):
                    failures.append(f"column sparsity m={m},q={q},d={d}")
                if not set(np.unique(a)).issubset({-1, 0, 1}):
                    failures.append(f"coefficients m={m},q={q},d={d}")
                for t in range(d):
                    rows = [r.row for r in records if r.t == t]
                    if len(rows) != len(set(rows)):
                        failures.append(f"duplicate rows m={m},q={q},d={d},t={t}")
    record("sparse_generator_structure", "PASS" if not failures else "FAIL", failures or "all tested shapes valid")


def check_unbiased_formula() -> None:
    rng = np.random.default_rng(20260928)
    v = rng.normal(size=20)
    truth = float(v @ v)
    vals = []
    for i in range(2000):
        _, a = sparse_matrix_gen(f"audit-unbiased-{i}".encode(), 30, 20, 4)
        vals.append(float(np.sum((a @ v) ** 2) / 4))
    mean = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
    z = abs(mean - truth) / se
    record("unbiased_squared_norm", "PASS" if z < 6 else "FAIL", {"truth": truth, "mean": mean, "se": se, "z": z})


def check_quantization_bounds() -> None:
    rng = np.random.default_rng(7)
    worst_vec_ratio = worst_pair_ratio = worst_sq_ratio = 0.0
    for d in (1, 4, 31, 128):
        for bits in (8, 12, 16):
            s = 1 << bits
            for _ in range(30):
                u, v = rng.uniform(-0.25, 0.25, size=(2, d))
                uq, vq = np.round(u * s) / s, np.round(v * s) / s
                vec_err = np.linalg.norm(uq - u)
                pair_err = abs(np.linalg.norm(uq - vq) - np.linalg.norm(u - v))
                sq_err = abs(np.sum((uq - vq) ** 2) - np.sum((u - v) ** 2))
                worst_vec_ratio = max(worst_vec_ratio, vec_err / quantize_error_bound(d, s))
                worst_pair_ratio = max(worst_pair_ratio, pair_err / pairwise_distance_perturbation_bound(d, s))
                denom = squared_distance_error_bound(float(np.linalg.norm(u - v)), d, s)
                worst_sq_ratio = max(worst_sq_ratio, sq_err / denom)
    ok = max(worst_vec_ratio, worst_pair_ratio, worst_sq_ratio) <= 1 + 1e-10
    record("quantization_error_bounds", "PASS" if ok else "FAIL", {
        "max_vector_bound_ratio": worst_vec_ratio,
        "max_pair_bound_ratio": worst_pair_ratio,
        "max_squared_bound_ratio": worst_sq_ratio,
    })


def check_no_wrap_formula() -> None:
    qp = QuantizationParams(S=1 << 16, B=4096, m=64, d=1024, n=1000)
    expected_dmax = 4 * qp.m * qp.d**2 * qp.B**2
    expected_psi = qp.n * expected_dmax
    formula_ok = qp.D_max == expected_dmax and qp.Psi_max == expected_psi and qp.p > 2 * qp.M
    record("no_wrap_formula_host", "PASS" if formula_ok else "FAIL", {
        "D_max": qp.D_max, "Psi_max": qp.Psi_max, "M": qp.M, "field": BN254_FR,
    }, "This validates the host formula, not enforcement of every bound inside the circuit.")


def check_certificate_formulas() -> None:
    eps = 0.15
    threshold = (1 + eps) / (1 - eps)
    x = np.array([[0.0, 0.0], [0.95, 0.0]])
    c = np.array([[0.0, 0.0], [2.0, 0.0]])
    cert = centroid_margin_certificate(x, c, eps)
    band = dbscan_ambiguity_band(1.0, eps)
    cb = cost_bounds(100.0, 100.0, eps)
    ok = bool(cert[0]) and (cert[1] == ((1.05**2 / 0.95**2) > threshold)) and cb["holds"]
    ok = ok and np.allclose(band, (1 / (1 + eps), 1 / (1 - eps)))
    record("certificate_formula_examples", "PASS" if ok else "FAIL", {
        "centroid_certificates": cert.tolist(), "dbscan_band": band, "cost_bounds": cb,
    })


def check_known_numeric_edge_cases() -> None:
    z = np.array([[0.9]])
    c = np.array([[0.1]])
    got_float = psi_hat(z, c, np.array([0]))
    expected_float = float(np.sum((z - c) ** 2))
    record("psi_hat_float_semantics", "PASS" if np.isclose(got_float, expected_float) else "FAIL", {
        "computed": got_float, "expected": expected_float,
    })

    z2 = np.array([[0]], dtype=np.int64)
    c2 = np.array([[3_100_000_000], [1]], dtype=np.int64)
    got_assign = nearest_centroid_assign(z2, c2).tolist()
    record("nearest_centroid_int64_overflow", "PASS" if got_assign == [1] else "FAIL", {
        "computed": got_assign, "expected": [1],
    })


def row_hash(row: dict) -> str:
    blob = json.dumps({k: row.get(k) for k in sorted(row)}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def check_result_files() -> None:
    summaries = {}
    from experiments.common import source_fingerprint
    problems = []
    current_source = source_fingerprint()
    for path in raw_result_paths():
        if not path.exists():
            problems.append(f"{path.name}: expected result file is missing")
            continue
        df = pd.read_parquet(path)
        summary = {"rows": len(df), "columns": len(df.columns)}
        if "_status" in df:
            summary["status"] = {str(k): int(v) for k, v in df["_status"].value_counts(dropna=False).items()}
        if "_hash" in df:
            summary["duplicate_hash_rows"] = int(df["_hash"].duplicated().sum())
        if df.empty or "_status" not in df or not df["_status"].isin(["ok", "native_only"]).all():
            problems.append(f"{path.name}: empty or unsuccessful rows")
        if summary.get("duplicate_hash_rows", 0):
            problems.append(f"{path.name}: duplicate row hashes")
        if "_source_sha256" not in df or not df["_source_sha256"].eq(current_source).all():
            problems.append(f"{path.name}: missing or stale source provenance")
        summaries[path.name] = summary
    record("result_file_integrity", "FAIL" if problems else "PASS" if summaries else "UNVERIFIED",
           {"files": summaries, "problems": problems})

    rq1_path = result_path("rq1_fidelity.parquet")
    if rq1_path.exists():
        d = pd.read_parquet(rq1_path)
        numeric_failures = {
            "ari_outside_range": int(((d.ari < -1) | (d.ari > 1)).sum()),
            "nmi_outside_range": int(((d.nmi < 0) | (d.nmi > 1)).sum()),
            "agreement_outside_range": int(((d.agreement < 0) | (d.agreement > 1)).sum()),
            "negative_costs": int(((d.phi_full < 0) | (d.phi_proj < 0) | (d.psi_hat < 0)).sum()),
            "r_phi_mismatch": int((~np.isclose(d.r_phi, d.phi_proj / d.phi_full, equal_nan=True)).sum()),
        }
        record("rq1_numeric_invariants", "PASS" if sum(numeric_failures.values()) == 0 else "FAIL", numeric_failures)


def regenerate_synthetic_hashes() -> None:
    from data.loaders import load_dataset

    path = result_path("rq1_fidelity.parquet")
    if not path.exists():
        record("synthetic_dataset_hashes", "UNVERIFIED", "rq1 result absent")
        return
    df = pd.read_parquet(path)
    tested, mismatches, errors = 0, [], []
    seen = set()
    for _, row in df.iterrows():
        spec = row.get("dataset")
        if not isinstance(spec, dict) or spec.get("loader") not in {"gaussian_mixture", "dbscan_margin"}:
            continue
        loader_fn = __import__("data.loaders", fromlist=["LOADERS"]).LOADERS[spec["loader"]]
        allowed = set(inspect.signature(loader_fn).parameters)
        kwargs = {k: v for k, v in dict(spec.get("kwargs", {})).items()
                  if k in allowed and v is not None
                  and not (isinstance(v, float) and np.isnan(v))}
        # Arrow's unified nested schema can promote integer configuration
        # fields to floats when other dataset loaders have null values.
        for k in ("n", "d", "k", "n_max", "seed"):
            if k in kwargs:
                kwargs[k] = int(kwargs[k])
        if "seed" not in kwargs:
            kwargs["seed"] = int(row.get("seed", 0))
        expected = str(row.get("dataset_sha256"))
        key = json.dumps({"loader": spec["loader"], "kwargs": kwargs,
                          "stored_hash": expected}, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        try:
            obj = load_dataset(spec["loader"], **kwargs)
            got = obj["meta"]["sha256"]
            tested += 1
            if got != expected:
                mismatches.append({"spec": spec, "seed": kwargs["seed"], "stored": expected, "regenerated": got})
        except Exception as exc:
            errors.append({"spec": spec, "error": repr(exc)})
    status = "PASS" if tested and not mismatches and not errors else "FAIL" if mismatches else "PARTIAL"
    record("synthetic_dataset_hashes", status, {"unique_regenerations": tested, "mismatches": mismatches, "errors": errors})


def check_real_data_provenance() -> None:
    cache = CODE / ".cache" / "commitjl_data"
    manifest_path = cache / "manifest.json"
    manifest = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as exc:
            record("real_dataset_manifest", "FAIL", repr(exc))
            return
    rq1 = result_path("rq1_fidelity.parquet")
    stored = {}
    if rq1.exists():
        d = pd.read_parquet(rq1)
        if {"dataset_name", "dataset_sha256"} <= set(d.columns):
            synth = {"gaussian_mixture", "dbscan_margin"}
            for name, group in d[~d.dataset_name.isin(synth)].groupby("dataset_name"):
                stored[str(name)] = sorted(str(x) for x in group.dataset_sha256.dropna().unique())
    # Manifest presence is provenance, not independent byte-level validation.
    status = "UNVERIFIED"
    record("real_dataset_manifest", status, {
        "cache_exists": cache.exists(), "manifest": manifest, "hashes_only_in_result_rows": stored,
    }, "A hash stored in a result row is not an independent download-integrity check.")


def check_configs_against_results() -> None:
    light = yaml.safe_load(PROOF_CONFIG.read_text())
    rq2_path = result_path("rq2_proof_cost.parquet")
    rq2 = pd.read_parquet(rq2_path) if rq2_path.exists() else pd.DataFrame()
    evidence = {
        "configured_light_reps": light.get("fixed", {}).get("n_reps_proof"),
        "stored_rq2_rows": len(rq2),
        "stored_measured_reps": sorted(rq2.n_reps.unique().tolist()) if "n_reps" in rq2 else [],
        "stored_variants": sorted(rq2.variant.unique().tolist()) if "variant" in rq2 else [],
    }
    reduced = rq2[(rq2.get("m", 0) < rq2.get("d", 0)) & (rq2.get("_status") == "ok")]
    evidence["successful_reduced_rows"] = len(reduced)
    evidence["all_statuses"] = {str(k): int(v) for k, v in rq2.get("_status", pd.Series(dtype=str)).value_counts().items()}
    ok = (len(rq2) >= 1 and len(reduced) > 0 and
          all(x == light.get("fixed", {}).get("n_reps_proof") for x in evidence["stored_measured_reps"]) and
          set(evidence["all_statuses"]) == {"ok"})
    record("proof_grid_real_dimensionality_reduction", "PASS" if ok else "FAIL", evidence)


def check_binary_platform() -> None:
    from commitjl.zk_bridge import locate_binary
    binary = locate_binary()
    magic = binary.read_bytes()[:4].hex() if binary.exists() else None
    # cffaedfe/feedfacf are Mach-O; 4d5a is Windows PE; 7f454c46 is ELF.
    if magic and magic.startswith("4d5a"):
        kind = "PE"
    else:
        kind = {"cffaedfe": "Mach-O", "feedfacf": "Mach-O", "7f454c46": "ELF"}.get(magic, "unknown")
    compatible = (platform.system() == "Darwin" and kind == "Mach-O") or (platform.system() == "Windows" and kind == "PE") or (platform.system() == "Linux" and kind == "ELF")
    record("committed_zk_binary_platform", "PASS" if compatible else "UNVERIFIED", {
        "host": platform.platform(), "binary_magic": magic, "binary_kind": kind,
    })


def check_circuit_and_bridge() -> None:
    go_test = subprocess.run(["go", "test", "-count=1", "./..."], cwd=CODE / "zk",
                             capture_output=True, text=True, timeout=120)
    record("zk_circuit_range_and_index_regressions", "PASS" if go_test.returncode == 0 else "FAIL", {
        "returncode": go_test.returncode, "output": (go_test.stdout + go_test.stderr)[-2000:],
    })

    proof = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_zk.py::test_protocol_transcript_proves_and_verifies_end_to_end", "-q"],
        cwd=CODE, capture_output=True, text=True, timeout=180,
    )
    record("protocol_transcript_to_proof_bridge", "PASS" if proof.returncode == 0 else "FAIL", {
        "returncode": proof.returncode, "output": (proof.stdout + proof.stderr)[-2000:],
    })


def check_dbscan_benchmark_and_q_normalization() -> None:
    from data.loaders import DBSCAN_MARGIN_BENCHMARK, load_dataset

    generated = load_dataset("dbscan_margin", **DBSCAN_MARGIN_BENCHMARK["kwargs"])["meta"]["sha256"]
    expected = DBSCAN_MARGIN_BENCHMARK["sha256"]
    record("dbscan_benchmark_hash", "PASS" if generated == expected else "FAIL",
           {"expected": expected, "generated": generated})

    check = subprocess.run([sys.executable, "-m", "pytest", "-q",
        "tests/test_numeric_and_data_regressions.py::test_rq4_normalized_sparse_projection_is_not_divided_by_q_twice"],
        cwd=CODE, capture_output=True, text=True, timeout=120)
    record("certificate_single_q_normalization", "PASS" if check.returncode == 0 else "FAIL",
           check.stdout + check.stderr)


def main() -> None:
    global RESULTS, PROOF_CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--proof-config", type=Path, default=PROOF_CONFIG)
    parser.add_argument("--out", type=Path, default=CODE / "validation/outputs/audit_results.json")
    args = parser.parse_args()
    RESULTS, PROOF_CONFIG = args.results_dir, args.proof_config
    check_sparse_generator()
    check_unbiased_formula()
    check_quantization_bounds()
    check_no_wrap_formula()
    check_certificate_formulas()
    check_known_numeric_edge_cases()
    check_result_files()
    regenerate_synthetic_hashes()
    check_real_data_provenance()
    check_configs_against_results()
    check_binary_platform()
    check_circuit_and_bridge()
    check_dbscan_benchmark_and_q_normalization()
    payload = {
        "audit_version": 2,
        "python": sys.version,
        "platform": platform.platform(),
        "summary": {s: sum(c["status"] == s for c in checks) for s in ("PASS", "FAIL", "PARTIAL", "UNVERIFIED")},
        "checks": checks,
    }
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    for c in checks:
        print(f"{c['status']:10} {c['name']}")
    if any(c["status"] != "PASS" for c in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

