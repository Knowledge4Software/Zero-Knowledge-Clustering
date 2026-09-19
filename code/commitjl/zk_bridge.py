"""Bridge between the Python protocol/experiment layer and the Go/gnark
circuit CLI (`zk/cmd/commitjl-prove`).

Responsibilities (plan P3):
  (a) serialize a Python-side instance into the instance JSON the Go CLI
      expects (field elements as decimal strings, matching encoding.py's
      canonical conventions where applicable);
  (b) invoke the built binary as a subprocess with a configurable timeout;
  (c) measure the process tree's peak RSS via psutil as a cross-check/
      fallback for the Go-reported RSS;
  (d) capture timeouts / OOM kills as structured failure records instead of
      letting exceptions bubble up, so experiment runners can log failures
      and continue.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is in requirements.txt
    psutil = None

from .encoding import SparseRecord
from .hashing import BN254_FR

ZK_DIR = Path(__file__).resolve().parent.parent / "zk"
BIN_PATH = ZK_DIR / "bin" / "commitjl-prove"


def _fe_str(x: int, p: int = BN254_FR) -> str:
    return str(int(x) % p)


@dataclass
class ZKInstance:
    """Python-side description of a circuit instance, mirroring
    zk/cmd/commitjl-prove's `Instance` JSON struct."""
    n: int
    d: int
    m: int
    k: int
    q: int
    variant: str
    backend: str  # "groth16" | "plonk"
    c_x: int
    c_theta: int
    c_a: int
    psi_hat: int
    centroids: np.ndarray       # k x d (field-encoded ints, signed via signed_to_field)
    X: np.ndarray                # n x d field-encoded ints
    r_x: int
    assign: List[int]
    sparse_records: Optional[List[SparseRecord]] = None  # for sparse/incircuit-gen
    dense_A: Optional[np.ndarray] = None                  # for dense/coord
    constant_A: Optional[np.ndarray] = None                # for pca-public/pisot/prepublished/full
    no_range_batch: bool = False
    range_bits: int = 48

    def to_json_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "n": self.n, "d": self.d, "m": self.m, "k": self.k, "q": self.q,
            "variant": self.variant,
            "backend": self.backend,
            "no_range_batch": self.no_range_batch,
            "range_bits": self.range_bits,
            "c_x": _fe_str(self.c_x),
            "c_theta": _fe_str(self.c_theta),
            "c_a": _fe_str(self.c_a),
            "psi_hat": _fe_str(self.psi_hat),
            "centroids": [[_fe_str(v) for v in row] for row in self.centroids],
            "x": [[_fe_str(v) for v in row] for row in self.X],
            "r_x": _fe_str(self.r_x),
            "assign": [int(a) for a in self.assign],
        }
        if self.sparse_records is not None:
            rows = [[0] * self.q for _ in range(self.d)]
            signs = [[0] * self.q for _ in range(self.d)]
            for r in self.sparse_records:
                rows[r.t][r.ell] = r.row
                signs[r.t][r.ell] = 1 if r.sign == 1 else BN254_FR - 1
            out["sparse_rows"] = [[_fe_str(v) for v in row] for row in rows]
            out["sparse_signs"] = [[_fe_str(v) for v in row] for row in signs]
        if self.dense_A is not None:
            out["dense_a"] = [[_fe_str(v) for v in row] for row in np.asarray(self.dense_A).tolist()]
        if self.constant_A is not None:
            out["constant_a"] = [[str(int(v)) for v in row] for row in np.asarray(self.constant_A).tolist()]
        return out


@dataclass
class ProveResult:
    ok: bool
    variant: str
    backend: str
    num_constraints: int = 0
    compile_seconds: float = 0.0
    setup_seconds: float = 0.0
    witness_seconds: float = 0.0
    prove_seconds: float = 0.0
    verify_seconds: float = 0.0
    proof_bytes: int = 0
    verify_ok: bool = False
    peak_rss_bytes_go: int = 0
    peak_rss_bytes_os: int = 0
    peak_rss_bytes_psutil: int = 0
    error: str = ""
    failure_kind: str = ""  # "", "timeout", "oom", "nonzero_exit", "parse_error"
    raw: Dict[str, Any] = field(default_factory=dict)


def locate_binary(build_if_missing: bool = True) -> Path:
    """Find (and optionally build) the commitjl-prove binary."""
    if BIN_PATH.exists():
        return BIN_PATH
    if not build_if_missing:
        raise FileNotFoundError(f"{BIN_PATH} not found and build_if_missing=False")
    BIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["go", "build", "-o", str(BIN_PATH), "./cmd/commitjl-prove"],
        cwd=str(ZK_DIR), capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"go build failed: {proc.stdout}\n{proc.stderr}")
    return BIN_PATH


def binary_available() -> bool:
    """True if the binary exists, or `go` is on PATH so it could be built."""
    if BIN_PATH.exists():
        return True
    import shutil
    return shutil.which("go") is not None


def _psutil_peak_rss(proc: subprocess.Popen, poll_interval: float = 0.02) -> Dict[str, int]:
    """Poll the process tree's RSS while `proc` runs; returns {'peak_rss': N}.
    Runs in a background thread started by the caller; safe to call from
    that thread directly."""
    peak = 0
    if psutil is None:
        return {"peak_rss": 0}
    try:
        p = psutil.Process(proc.pid)
    except Exception:
        return {"peak_rss": 0}
    while proc.poll() is None:
        try:
            procs = [p] + p.children(recursive=True)
            total = 0
            for pr in procs:
                try:
                    total += pr.memory_info().rss
                except Exception:
                    continue
            peak = max(peak, total)
        except Exception:
            pass
        time.sleep(poll_interval)
    return {"peak_rss": peak}


def run_prove(instance: ZKInstance, timeout: float = 300.0, build_if_missing: bool = True) -> ProveResult:
    """Invoke the commitjl-prove binary on `instance`, returning a
    ProveResult. Timeouts and abnormal termination (e.g. OOM kill, signalled
    process) are captured as structured failures rather than raised."""
    try:
        binpath = locate_binary(build_if_missing=build_if_missing)
    except Exception as e:
        return ProveResult(ok=False, variant=instance.variant, backend=instance.backend,
                            error=str(e), failure_kind="build_failed")

    payload = json.dumps(instance.to_json_dict())

    try:
        proc = subprocess.Popen(
            [str(binpath)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as e:
        return ProveResult(ok=False, variant=instance.variant, backend=instance.backend,
                            error=str(e), failure_kind="spawn_failed")

    rss_holder: Dict[str, int] = {"peak_rss": 0}

    def _watch():
        rss_holder.update(_psutil_peak_rss(proc))

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()

    try:
        stdout, stderr = proc.communicate(input=payload, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        watcher.join(timeout=2)
        return ProveResult(ok=False, variant=instance.variant, backend=instance.backend,
                            error=f"proving timed out after {timeout}s", failure_kind="timeout",
                            peak_rss_bytes_psutil=rss_holder.get("peak_rss", 0))

    watcher.join(timeout=2)

    returncode = proc.returncode
    # negative returncode on POSIX means killed by signal (e.g. SIGKILL=-9,
    # typically an OOM kill).
    if returncode is not None and returncode < 0:
        return ProveResult(ok=False, variant=instance.variant, backend=instance.backend,
                            error=f"process killed by signal {-returncode} (possible OOM)",
                            failure_kind="oom",
                            peak_rss_bytes_psutil=rss_holder.get("peak_rss", 0))

    try:
        raw = json.loads(stdout)
    except Exception:
        return ProveResult(ok=False, variant=instance.variant, backend=instance.backend,
                            error=f"failed to parse result JSON: {stderr[-2000:]}",
                            failure_kind="parse_error",
                            peak_rss_bytes_psutil=rss_holder.get("peak_rss", 0))

    res = ProveResult(
        ok=(returncode == 0 and not raw.get("error")),
        variant=raw.get("variant", instance.variant),
        backend=raw.get("backend", instance.backend),
        num_constraints=raw.get("num_constraints", 0),
        compile_seconds=raw.get("compile_seconds", 0.0),
        setup_seconds=raw.get("setup_seconds", 0.0),
        witness_seconds=raw.get("witness_seconds", 0.0),
        prove_seconds=raw.get("prove_seconds", 0.0),
        verify_seconds=raw.get("verify_seconds", 0.0),
        proof_bytes=raw.get("proof_bytes", 0),
        verify_ok=raw.get("verify_ok", False),
        peak_rss_bytes_go=raw.get("peak_rss_bytes_go_memstats", 0),
        peak_rss_bytes_os=raw.get("peak_rss_bytes_os_maxrss", 0),
        peak_rss_bytes_psutil=rss_holder.get("peak_rss", 0),
        error=raw.get("error", "") if not res_ok(returncode, raw) else "",
        failure_kind="" if res_ok(returncode, raw) else "prove_or_verify_failed",
        raw=raw,
    )
    return res


def res_ok(returncode: int, raw: Dict[str, Any]) -> bool:
    return returncode == 0 and not raw.get("error")


def field_hash_theta(centroids: np.ndarray, p: int = BN254_FR) -> int:
    """Circuit-friendly C_theta: MiMC hash over the raw sequence of
    field-encoded centroid coordinates (row-major), unlike commit.py's
    commit_theta which hashes a byte-packed encoding (bit-level packing
    that would be expensive to replicate in-circuit). This is a documented
    scope reduction for the ZK bridge (P3): C_theta as authenticated by the
    circuit is this field-native digest, not commit.py's byte-encoded one.
    C_X (the Merkle root) is unaffected and matches commit.py exactly
    (verified in tests/test_hashing.py's golden-vector tests), since
    commit_matrix's leaf/node hashing is already field-native."""
    from .encoding import signed_to_field
    from .hashing import mimc_hash
    elems = [signed_to_field(int(v), p) for v in np.asarray(centroids).astype(np.int64).ravel(order="C")]
    return mimc_hash(elems, p)


def field_hash_sparse_list(records: List[SparseRecord], p: int = BN254_FR) -> int:
    """Circuit-friendly C_A: MiMC hash over the raw sequence of (row, sign)
    field elements in (t asc, ell asc) order -- see field_hash_theta's
    docstring for why this differs from commit.py's byte-packed
    commit_sparse_matrix."""
    from .hashing import mimc_hash
    elems: List[int] = []
    for r in records:
        elems.append(r.row % p)
        elems.append(1 if r.sign == 1 else p - 1)
    return mimc_hash(elems, p)


def build_sparse_instance(X: np.ndarray, centroids: np.ndarray, seed: bytes, q: int, m: int,
                           backend: str = "groth16", range_bits: int = 48,
                           tamper: Optional[str] = None) -> "ZKInstance":
    """Build a full ZKInstance for the 'sparse' variant from native numpy
    inputs, wiring together sparse_gen.py, commit.py, clustering.py and
    encoding.py exactly as the Python protocol would, for use by
    tests/test_zk.py. `tamper` injects a specific negative-test corruption:
    one of {None, "wrong_A", "wrong_X_open", "wrong_assign", "tie_break",
    "wrong_psi", "wraparound"}.

    NOTE: C_theta and C_A use field_hash_theta/field_hash_sparse_list (a
    circuit-friendly, field-native digest), not commit.py's byte-packed
    commit_theta/commit_sparse_matrix -- see those functions' docstrings.
    C_X (Merkle root) is exactly commit.py's commit_matrix.
    """
    from .commit import commit_matrix
    from .sparse_gen import sparse_matrix_gen
    from .encoding import signed_to_field
    from .clustering import nearest_centroid_assign, psi_hat as psi_hat_fn

    n, d = X.shape
    k = centroids.shape[0]

    records, A = sparse_matrix_gen(seed, m, d, q)

    Z = (A @ X.T).T          # n x m
    Cz = (A @ centroids.T).T  # k x m

    if tamper == "tie_break":
        # force a tie between centroid 0 and 1 for point 0 by making them
        # equidistant, then assert (incorrectly) assignment to centroid 1.
        Cz = Cz.copy()
        Cz[1] = Cz[0] + (Cz[0] - Z[0]) * 0  # keep centroid1 == centroid0's projection is too strong; instead:
        Cz[1] = Cz[0].copy()

    assign = nearest_centroid_assign(Z, Cz)
    psi = psi_hat_fn(Z, Cz, assign)

    if tamper == "wrong_assign":
        assign = assign.copy()
        assign[0] = (assign[0] + 1) % k
    if tamper == "tie_break":
        assign = assign.copy()
        assign[0] = 1  # violates lowest-index tie-break (should be 0)
    if tamper == "wrong_psi":
        psi = psi + 1

    Xq = X.astype(np.int64)
    if tamper == "wraparound":
        Xq = Xq.copy()
        Xq[0, 0] = Xq[0, 0] + (1 << 40)  # force an out-of-bound quantized value

    merkle = commit_matrix(Xq)
    c_x = merkle.root
    c_theta = field_hash_theta(centroids.astype(np.int64))
    c_a = field_hash_sparse_list(records)

    if tamper == "wrong_A":
        # tamper one sparse record's sign so hash(A) != C_A but keep the
        # projection consistent with the (correct) C_A, i.e. this must fail
        # the hash(sparse list)==C_A check.
        records = list(records)
        first = records[0]
        from .encoding import SparseRecord
        records[0] = SparseRecord(t=first.t, ell=first.ell, row=first.row, sign=-first.sign)
    if tamper == "wrong_X_open":
        c_x = (c_x + 1) % BN254_FR

    centroids_field = [[signed_to_field(int(v), BN254_FR) for v in row] for row in centroids.astype(np.int64).tolist()]
    X_field = [[signed_to_field(int(v), BN254_FR) for v in row] for row in Xq.tolist()]

    inst = ZKInstance(
        n=n, d=d, m=m, k=k, q=q,
        variant="sparse", backend=backend,
        c_x=c_x, c_theta=c_theta, c_a=c_a, psi_hat=int(psi),
        centroids=centroids_field, X=X_field, r_x=merkle.salt,
        assign=[int(a) for a in assign],
        sparse_records=records,
        range_bits=range_bits,
    )
    return inst
