"""End-to-end Algorithm 1 (native/host-side reference), with per-stage timing
and byte-size instrumentation into a transcript dict/JSON.

Manuscript reference: Fig./Alg. 1 "Commit-then-sample sparse projection
protocol":
1. Input: private integer X; public mu; public theta=(C, m, q).
2. (C_X, r_X) <- Commit(Encode(X, mu)).
3. Publish (C_X, C_theta) before learning rho.
4. s <- H("CommitJL-v1" || C_X || C_theta || rho).
5. Verifier derives A <- SparseMatrixGen(s, m, d, q) and C_A <- H_A(Encode(A)).
6. Z <- X A^T (signed integer semantics).
7. y <- Cluster(Z, theta).
8. pi <- Prove(...).  [P3, out of scope here: we stub a native "proof" marker]
9. Return (C_X, C_theta, C_A, rho, theta, y, pi).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .commit import commit_matrix, commit_theta, commit_sparse_matrix
from .encoding import VERSION_TAG, encode_sparse_list, dense_to_sparse_list
from .hashing import BN254_FR, mimc_hash_bytes
from .randomness import TaskLog
from .sparse_gen import sparse_matrix_gen
from .clustering import nearest_centroid_assign, psi_hat


def _u32le(x: int) -> bytes:
    return int(x).to_bytes(4, "little")


def derive_seed(C_X: int, C_theta: int, rho: bytes, p: int = BN254_FR) -> bytes:
    """s <- H("CommitJL-v1" || C_X || C_theta || rho). Returns raw bytes
    (used directly as the SHAKE256 seed for SparseMatrixGen)."""
    blob = VERSION_TAG + C_X.to_bytes(32, "big") + C_theta.to_bytes(32, "big") + rho
    # Use SHA-256 as H here (host-side seed derivation hash; distinct role
    # from the in-circuit MiMC commitments C_X/C_theta/C_A).
    import hashlib
    return hashlib.sha256(blob).digest()


@dataclass
class Transcript:
    stage_times_s: Dict[str, float] = field(default_factory=dict)
    stage_bytes: Dict[str, int] = field(default_factory=dict)
    public: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "stage_times_s": self.stage_times_s,
            "stage_bytes": self.stage_bytes,
            "public": self.public,
        }


class Prover:
    def __init__(self, X: np.ndarray, scale: int, prover_id: str = "prover0"):
        self.X = np.asarray(X)
        self.scale = scale
        self.prover_id = prover_id
        self.r_X = None
        self.commitment = None

    def commit_data(self):
        t0 = time.perf_counter()
        self.commitment = commit_matrix(self.X)
        self.r_X = self.commitment.salt
        dt = time.perf_counter() - t0
        return self.commitment.root, dt

    def project_and_cluster(self, A_dense: np.ndarray, centroids: np.ndarray):
        t0 = time.perf_counter()
        Z = self.X @ A_dense.T  # signed integer semantics
        assignments = nearest_centroid_assign(Z, (centroids @ A_dense.T))
        Psi_hat = psi_hat(Z, (centroids @ A_dense.T), assignments)
        dt = time.perf_counter() - t0
        return Z, assignments, Psi_hat, dt


class Verifier:
    def __init__(self, task_log: TaskLog, m: int, q: int, centroids: np.ndarray, mu: bytes = b""):
        self.task_log = task_log
        self.m = m
        self.q = q
        self.centroids = np.asarray(centroids)
        self.mu = mu
        self.C_theta = None

    def record_theta(self):
        t0 = time.perf_counter()
        self.C_theta = commit_theta(self.centroids, self.m, self.q, self.mu)
        dt = time.perf_counter() - t0
        return self.C_theta, dt

    def issue_nonce(self):
        return self.task_log.fresh_nonce(), 0.0

    def expand_sparse(self, C_X: int, rho: bytes, d: int):
        t0 = time.perf_counter()
        seed = derive_seed(C_X, self.C_theta, rho)
        records, A_dense = sparse_matrix_gen(seed, self.m, d, self.q)
        C_A = commit_sparse_matrix(A_dense)
        dt = time.perf_counter() - t0
        nbytes = len(encode_sparse_list(records))
        return seed, records, A_dense, C_A, dt, nbytes


def run_protocol(X: np.ndarray, centroids: np.ndarray, m: int, q: int, scale: int,
                  task_log: Optional[TaskLog] = None, mu: bytes = b"",
                  prover_id: str = "prover0") -> Dict[str, Any]:
    """Runs Algorithm 1 end-to-end natively and returns a transcript dict."""
    n, d = X.shape
    if task_log is None:
        task_log = TaskLog(Q=1)

    transcript = Transcript()
    prover = Prover(X, scale, prover_id)
    verifier = Verifier(task_log, m, q, centroids, mu)

    C_X, dt_commit = prover.commit_data()
    transcript.stage_times_s["commit_X"] = dt_commit
    transcript.stage_bytes["commit_X"] = 32  # digest size

    C_theta, dt_theta = verifier.record_theta()
    transcript.stage_times_s["record_theta"] = dt_theta
    transcript.stage_bytes["record_theta"] = 32

    task_log.record_commitment(prover_id, C_X, C_theta)

    t0 = time.perf_counter()
    rho, _ = verifier.issue_nonce()
    transcript.stage_times_s["issue_nonce"] = time.perf_counter() - t0
    transcript.stage_bytes["issue_nonce"] = len(rho)

    seed, records, A_dense, C_A, dt_expand, sparse_bytes = verifier.expand_sparse(C_X, rho, d)
    transcript.stage_times_s["expand_sparse_and_CA"] = dt_expand
    transcript.stage_bytes["expand_sparse_and_CA"] = sparse_bytes + 32

    Z, assignments, Psi_hat, dt_cluster = prover.project_and_cluster(A_dense, centroids)
    transcript.stage_times_s["project_and_cluster"] = dt_cluster
    transcript.stage_bytes["project_and_cluster"] = int(Z.nbytes)

    # P3 (ZK proof generation) is out of scope; record a native-only marker.
    transcript.public.update({
        "version": VERSION_TAG.decode(),
        "n": n, "d": d, "m": m, "q": q, "k": int(centroids.shape[0]),
        "C_X": C_X, "C_theta": C_theta, "C_A": C_A,
        "rho": rho.hex(),
        "Psi_hat": int(Psi_hat),
        "proof": "NATIVE_ONLY_NO_ZK_PROOF (P3 out of scope for this build)",
    })

    return {
        "transcript": transcript.to_json(),
        "assignments": assignments,
        "Psi_hat": int(Psi_hat),
        "A_dense": A_dense,
        "Z": Z,
        "C_X": C_X, "C_theta": C_theta, "C_A": C_A, "rho": rho,
    }
