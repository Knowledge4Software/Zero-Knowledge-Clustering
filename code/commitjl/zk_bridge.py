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
import os
import shlex
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
BIN_PATH = Path(os.environ.get("COMMITJL_ZK_BIN", ZK_DIR / "bin" / ("commitjl-prove.exe" if os.name == "nt" else "commitjl-prove")))


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
    projection_seed: Optional[bytes] = None  # host-verified transcript binding
    challenge_rho: Optional[bytes] = None   # serialized for audit; not a circuit input
    benchmark_metrics: Dict[str, Any] = field(default_factory=dict)  # host measurements only
    legacy_projection: bool = False  # diagnostic comparison with original routing
    legacy_bounds: bool = False      # diagnostic comparison with original bounds
    score_distances: bool = False
    lookup_sparse: bool = False  # generic authenticated sparse scatter
    specialize_sparse: bool = False  # fresh row pattern requires new circuit/setup
    protected_basis: Optional[np.ndarray] = None  # compiled public integer B
    basis_scale: int = 0
    solve_only: bool = False  # constraint solver diagnostic; no proof
    diagnostic_prefix: str = ""  # truncated relation, only allowed with solve_only

    def to_json_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "n": self.n, "d": self.d, "m": self.m, "k": self.k, "q": self.q,
            "variant": self.variant,
            "backend": self.backend,
            "no_range_batch": self.no_range_batch,
            "range_bits": self.range_bits,
            "legacy_projection": self.legacy_projection,
            "legacy_bounds": self.legacy_bounds,
            "score_distances": self.score_distances,
            "lookup_sparse": self.lookup_sparse,
            "specialize_sparse": self.specialize_sparse,
            "basis_scale": int(self.basis_scale),
            "solve_only": self.solve_only,
            "diagnostic_prefix": self.diagnostic_prefix,
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
        if self.protected_basis is not None:
            out['protected_basis'] = [[str(int(v)) for v in row] for row in np.asarray(self.protected_basis).tolist()]
        if self.projection_seed is not None:
            out["projection_seed"] = self.projection_seed.hex()
        if self.challenge_rho is not None:
            out["challenge_rho"] = self.challenge_rho.hex()
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
    repetitions: int = 1
    successful_proofs: int = 0
    peak_rss_bytes_go: int = 0
    peak_rss_bytes_os: int = 0
    peak_rss_bytes_psutil: int = 0
    error: str = ""
    failure_kind: str = ""  # "", "timeout", "oom", "nonzero_exit", "parse_error"
    raw: Dict[str, Any] = field(default_factory=dict)


def locate_binary(build_if_missing: bool = True) -> Path:
    """Find (and optionally build) the commitjl-prove binary."""
    sources = list(ZK_DIR.rglob("*.go")) + [ZK_DIR / "go.mod", ZK_DIR / "go.sum"]
    stale = BIN_PATH.exists() and any(p.stat().st_mtime > BIN_PATH.stat().st_mtime for p in sources)
    # An explicitly configured external runner/binary is managed by its owner.
    if BIN_PATH.exists() and (not stale or not build_if_missing or "COMMITJL_ZK_BIN" in os.environ):
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


def _binary_command(binpath: Path) -> List[str]:
    """Return the executable command, optionally through a configured runner.

    ``COMMITJL_ZK_RUNNER=wsl`` is useful for executing a Linux cross-compiled
    proof binary from a Windows Python environment. Other runner values are
    tokenized with shell-like quoting and receive the binary path as their
    final argument.
    """
    runner = os.environ.get("COMMITJL_ZK_RUNNER", "").strip()
    if not runner:
        return [str(binpath)]
    command = shlex.split(runner, posix=os.name != "nt")
    if command and Path(command[0]).name.lower() in {"wsl", "wsl.exe"} and os.name == "nt":
        resolved = str(binpath.resolve())
        drive, tail = os.path.splitdrive(resolved)
        if not drive:
            raise ValueError(f"WSL runner requires an absolute Windows binary path: {resolved}")
        linux_tail = tail.lstrip("\\/").replace("\\", "/")
        translated = f"/mnt/{drive[0].lower()}/{linux_tail}"
        return command + [translated]
    return command + [str(binpath)]


def _kill_process_tree(proc):
    if psutil is not None:
        try:
            children = psutil.Process(proc.pid).children(recursive=True)
            for child in children:
                try:
                    child.kill()
                except psutil.Error:
                    pass
        except psutil.Error:
            pass
    try:
        proc.kill()
    except OSError:
        pass


def _psutil_peak_rss(proc: subprocess.Popen, poll_interval: float = 0.02, max_rss_bytes=None) -> Dict[str, int]:
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
            if max_rss_bytes is not None and total > max_rss_bytes:
                _kill_process_tree(proc)
                return {"peak_rss": peak, "memory_limit_exceeded": 1}
        except Exception:
            pass
        time.sleep(poll_interval)
    return {"peak_rss": peak}


def run_prove(instance: ZKInstance, timeout: float = 300.0, build_if_missing: bool = True,
              repetitions: int = 1, warmup: int = 0, max_rss_bytes: Optional[int] = None) -> ProveResult:
    """Invoke the commitjl-prove binary on `instance`, returning a
    ProveResult. Timeouts and abnormal termination (e.g. OOM kill, signalled
    process) are captured as structured failures rather than raised."""
    try:
        binpath = locate_binary(build_if_missing=build_if_missing)
    except Exception as e:
        return ProveResult(ok=False, variant=instance.variant, backend=instance.backend,
                            error=str(e), failure_kind="build_failed")

    if repetitions < 1 or warmup < 0:
        raise ValueError("repetitions must be >= 1 and warmup must be >= 0")
    payload_dict = instance.to_json_dict()
    payload_dict.update({"n_repetitions": repetitions, "n_warmup": warmup})
    payload = json.dumps(payload_dict)

    try:
        proc = subprocess.Popen(
            _binary_command(binpath),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as e:
        return ProveResult(ok=False, variant=instance.variant, backend=instance.backend,
                            error=str(e), failure_kind="spawn_failed")

    rss_holder: Dict[str, int] = {"peak_rss": 0}
    if max_rss_bytes is None and psutil is not None:
        memory = psutil.virtual_memory()
        max_rss_bytes = int(os.environ.get("COMMITJL_MAX_PROOF_RSS_BYTES",
                            min(8 * 1024**3, memory.total * 0.55, memory.available * 0.8)))
    if max_rss_bytes is not None and max_rss_bytes <= 0:
        _kill_process_tree(proc)
        proc.communicate()
        raise ValueError("max_rss_bytes must be positive")

    def _watch():
        rss_holder.update(_psutil_peak_rss(proc, max_rss_bytes=max_rss_bytes))

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()

    try:
        stdout, stderr = proc.communicate(input=payload, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        watcher.join(timeout=2)
        return ProveResult(ok=False, variant=instance.variant, backend=instance.backend,
                            error=f"proving timed out after {timeout}s", failure_kind="timeout",
                            peak_rss_bytes_psutil=rss_holder.get("peak_rss", 0))
    except BaseException:
        _kill_process_tree(proc)
        proc.communicate()
        watcher.join(timeout=2)
        raise

    watcher.join(timeout=2)
    if rss_holder.get("memory_limit_exceeded"):
        return ProveResult(ok=False, variant=instance.variant, backend=instance.backend,
                           error=f"proof process exceeded RSS budget of {max_rss_bytes} bytes",
                           failure_kind="memory_limit", peak_rss_bytes_psutil=rss_holder["peak_rss"])

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

    operation_ok = ((returncode == 0 and raw.get('solve_only') is True and
                     raw.get('solve_ok') is True and not raw.get('error'))
                    if instance.solve_only else
                    (res_ok(returncode, raw) and raw.get('successful_proofs') == repetitions))
    res = ProveResult(
        ok=operation_ok,
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
        repetitions=raw.get("n_repetitions", repetitions),
        successful_proofs=raw.get("successful_proofs", 0),
        peak_rss_bytes_go=raw.get("peak_rss_bytes_go_memstats", 0),
        peak_rss_bytes_os=raw.get("peak_rss_bytes_os_maxrss", 0),
        peak_rss_bytes_psutil=rss_holder.get("peak_rss", 0),
        error=raw.get("error", "") if not operation_ok else "",
        failure_kind="" if operation_ok else ('constraint_solve_failed' if instance.solve_only else "prove_or_verify_failed"),
        raw=raw,
    )
    return res


def res_ok(returncode: int, raw: Dict[str, Any]) -> bool:
    return returncode == 0 and not raw.get("error") and raw.get("verify_ok") is True


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


def build_sparse_instance_from_transcript(
        X: np.ndarray, centroids: np.ndarray, *, seed: bytes, rho: bytes,
        records: List[SparseRecord], A_dense: np.ndarray, assignments: np.ndarray,
        psi_hat: int, c_x: int, r_x: int, c_theta: int, c_a: int,
        q: int, m: int, backend: str = "groth16", range_bits: int = 48) -> ZKInstance:
    """Build a ZK witness from an already-created protocol transcript.

    The bridge independently regenerates ``A`` from the transcript-derived
    seed and validates every commitment and native output before serializing
    the witness. This prevents experiments from substituting an unrelated
    seed or matrix between the protocol and proof layers.
    """
    from .clustering import nearest_centroid_assign, psi_hat as psi_hat_fn
    from .commit import commit_matrix
    from .encoding import signed_to_field
    from .sparse_gen import sparse_matrix_gen
    from .protocol import derive_seed

    X = np.asarray(X)
    centroids = np.asarray(centroids)
    A_dense = np.asarray(A_dense)
    assignments = np.asarray(assignments, dtype=np.int64)
    if not np.issubdtype(X.dtype, np.integer) or not np.issubdtype(centroids.dtype, np.integer):
        raise TypeError("ZK transcript inputs and centroids must be quantized integer arrays")
    if X.ndim != 2 or centroids.ndim != 2 or X.shape[1] != centroids.shape[1]:
        raise ValueError("incompatible transcript data and centroid shapes")
    n, d = X.shape
    k = centroids.shape[0]
    if A_dense.shape != (m, d):
        raise ValueError(f"projection matrix must have shape {(m, d)}, got {A_dense.shape}")
    if assignments.shape != (n,):
        raise ValueError(f"assignments must have shape {(n,)}, got {assignments.shape}")

    if seed != derive_seed(int(c_x), int(c_theta), rho):
        raise ValueError("projection seed does not match the commitment/challenge transcript")
    if np.any(X > 4096) or np.any(X < -4096) or np.any(centroids > 4096) or np.any(centroids < -4096):
        raise ValueError("circuit input coordinates must be in [-4096, 4096]")
    regenerated_records, regenerated_A = sparse_matrix_gen(seed, m, d, q)
    if list(records) != regenerated_records or not np.array_equal(A_dense, regenerated_A):
        raise ValueError("transcript matrix does not match the challenge-derived seed")

    merkle = commit_matrix(X.astype(np.int64), salt=int(r_x))
    if merkle.root != int(c_x):
        raise ValueError("transcript X/r_X does not open C_X")
    expected_theta = field_hash_theta(centroids)
    if expected_theta != int(c_theta):
        raise ValueError("transcript centroids do not match C_theta")
    expected_a = field_hash_sparse_list(regenerated_records)
    if expected_a != int(c_a):
        raise ValueError("challenge-derived sparse matrix does not match C_A")

    Z = (regenerated_A @ X.T).T
    projected_centroids = (regenerated_A @ centroids.T).T
    expected_assignments = nearest_centroid_assign(Z, projected_centroids)
    expected_psi = psi_hat_fn(Z, projected_centroids, expected_assignments)
    if not np.array_equal(assignments, expected_assignments):
        raise ValueError("transcript assignments do not match native argmin")
    if int(psi_hat) != int(expected_psi):
        raise ValueError("transcript Psi_hat does not match native projected cost")

    centroids_field = np.array([
        [signed_to_field(int(v), BN254_FR) for v in row]
        for row in centroids.tolist()
    ], dtype=object)
    x_field = np.array([
        [signed_to_field(int(v), BN254_FR) for v in row]
        for row in X.tolist()
    ], dtype=object)
    return ZKInstance(
        n=n, d=d, m=m, k=k, q=q, variant="sparse", backend=backend,
        c_x=int(c_x), c_theta=int(c_theta), c_a=int(c_a), psi_hat=int(psi_hat),
        centroids=centroids_field, X=x_field, r_x=int(r_x),
        assign=[int(a) for a in assignments], sparse_records=list(records),
        range_bits=range_bits, projection_seed=bytes(seed), challenge_rho=bytes(rho),
    )


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
