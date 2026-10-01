"""ZK circuit tests (plan P3). Skips the whole module if the Go binary isn't
built/available (and `go` isn't even on PATH to build it), so CI without Go
still passes.

Covers, for the 'sparse' variant:
  - an honest n=20,d=32,m=8,k=3,q=2 instance proves+verifies for groth16 AND
    plonk;
  - native Python Psi_hat (clustering.py) equals the circuit's public
    Psi_hat for that instance (by construction, since both are computed
    from the same nearest_centroid_assign/psi_hat call, and the circuit
    proving step re-derives/asserts the same value in-circuit -- a proof
    failure would indicate a mismatch);
  - six negative tests, each of which must fail to prove: wrong A (tampered
    sparse list vs C_A), wrong X Merkle opening, wrong cluster assignment,
    a tie-break violation, wrong Psi_hat, and a forced quantization
    wraparound value.
"""
from __future__ import annotations

import numpy as np
import pytest

from commitjl.zk_bridge import binary_available, build_sparse_instance, run_prove
from commitjl.clustering import nearest_centroid_assign, psi_hat as psi_hat_fn
from commitjl.protocol import derive_seed, run_protocol

pytestmark = pytest.mark.skipif(
    not binary_available(),
    reason="Go toolchain / commitjl-prove binary not available",
)

N, D, M, K, Q = 20, 32, 8, 3, 2
SEED = b"commitjl-test-zk-seed"


def _make_instance(backend: str, tamper=None):
    rng = np.random.default_rng(42)
    X = rng.integers(-50, 50, size=(N, D)).astype(np.int64)
    centroids = rng.integers(-50, 50, size=(K, D)).astype(np.int64)
    return build_sparse_instance(X, centroids, seed=SEED, q=Q, m=M, backend=backend,
                                  range_bits=48, tamper=tamper), X, centroids


@pytest.mark.parametrize("backend", ["groth16", "plonk"])
def test_honest_instance_proves_and_verifies(backend):
    inst, X, centroids = _make_instance(backend)
    res = run_prove(inst, timeout=300)
    assert res.ok, f"prove/verify failed: {res.error}"
    assert res.verify_ok
    assert res.num_constraints > 0
    assert res.proof_bytes > 0


def test_native_psi_hat_matches_instance_psi_hat():
    from commitjl.sparse_gen import sparse_matrix_gen
    rng = np.random.default_rng(42)
    X = rng.integers(-50, 50, size=(N, D)).astype(np.int64)
    centroids = rng.integers(-50, 50, size=(K, D)).astype(np.int64)
    records, A = sparse_matrix_gen(SEED, M, D, Q)
    Z = (A @ X.T).T
    Cz = (A @ centroids.T).T
    assign = nearest_centroid_assign(Z, Cz)
    native_psi = psi_hat_fn(Z, Cz, assign)

    inst, _, _ = _make_instance("groth16")
    assert inst.psi_hat == native_psi

    res = run_prove(inst, timeout=300)
    assert res.ok
    assert res.verify_ok


def test_protocol_transcript_proves_and_verifies_end_to_end():
    rng = np.random.default_rng(7)
    X = rng.integers(-8, 9, size=(4, 4), dtype=np.int64)
    centroids = rng.integers(-8, 9, size=(2, 4), dtype=np.int64)
    result = run_protocol(
        X, centroids, m=2, q=1, scale=256,
        generate_zk_proof=True, backend="groth16", proof_timeout=300,
    )

    proof = result["proof_result"]
    public = result["transcript"]["public"]
    assert result["seed"] == derive_seed(result["C_X"], result["C_theta"], result["rho"])
    assert proof is not None and proof.ok and proof.verify_ok, proof.error
    assert public["proof"]["ok"] is True
    assert public["proof"]["verify_ok"] is True
    assert proof.num_constraints > 0
    assert proof.proof_bytes > 0


@pytest.mark.parametrize("tamper", [
    "wrong_A", "wrong_X_open", "wrong_assign", "tie_break", "wrong_psi", "wraparound",
])
def test_negative_cases_fail(tamper):
    inst, _, _ = _make_instance("groth16", tamper=tamper)
    res = run_prove(inst, timeout=300)
    assert not res.ok, f"tamper={tamper} unexpectedly proved/verified"
    assert not res.verify_ok
