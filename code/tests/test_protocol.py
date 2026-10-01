import numpy as np

from commitjl.protocol import derive_seed, run_protocol
from commitjl.randomness import TaskLog
from commitjl.zk_bridge import ProveResult, field_hash_sparse_list, field_hash_theta


def _small_instance(seed=0):
    rng = np.random.default_rng(seed)
    n, d, k = 12, 16, 3
    X = rng.integers(-50, 50, size=(n, d)).astype(np.int64)
    centroids = rng.integers(-50, 50, size=(k, d)).astype(np.int64)
    return X, centroids


def test_protocol_runs_end_to_end():
    X, centroids = _small_instance()
    result = run_protocol(X, centroids, m=8, q=2, scale=256)
    assert "Psi_hat" in result
    assert result["Psi_hat"] >= 0
    assert result["assignments"].shape[0] == X.shape[0]
    assert result["A_dense"].shape == (8, 16)


def test_protocol_deterministic_given_fixed_rho():
    """With a fixed rho (bypassing the TaskLog's fresh-nonce randomness), the
    sparse matrix expansion and resulting Psi_hat must be identical across
    repeated runs, since SparseMatrixGen is a deterministic function of s."""
    X, centroids = _small_instance()
    from commitjl.commit import commit_matrix
    from commitjl.sparse_gen import sparse_matrix_gen
    from commitjl.clustering import nearest_centroid_assign, psi_hat

    commitment = commit_matrix(X, salt=12345)
    C_X = commitment.root
    C_theta = field_hash_theta(centroids)
    rho = b"\x00" * 32

    def run_once():
        seed = derive_seed(C_X, C_theta, rho)
        records, A = sparse_matrix_gen(seed, m=8, d=X.shape[1], q=2)
        Z = X @ A.T
        Cp = centroids @ A.T
        a = nearest_centroid_assign(Z, Cp)
        return psi_hat(Z, Cp, a), A

    psi1, A1 = run_once()
    psi2, A2 = run_once()
    assert psi1 == psi2
    assert np.array_equal(A1, A2)


def test_protocol_proof_path_uses_exact_transcript(monkeypatch):
    X, centroids = _small_instance()
    captured = {}

    def fake_run_prove(instance, timeout, build_if_missing):
        captured["instance"] = instance
        return ProveResult(ok=True, variant="sparse", backend=instance.backend,
                           num_constraints=123, proof_bytes=196, verify_ok=True)

    monkeypatch.setattr("commitjl.protocol.run_prove", fake_run_prove)
    result = run_protocol(
        X, centroids, m=8, q=2, scale=256, generate_zk_proof=True,
        backend="groth16", build_if_missing=False,
    )
    instance = captured["instance"]
    public = result["transcript"]["public"]

    assert result["seed"] == derive_seed(result["C_X"], result["C_theta"], result["rho"])
    assert result["C_theta"] == field_hash_theta(centroids)
    assert result["C_A"] == field_hash_sparse_list(result["records"])
    assert instance.projection_seed == result["seed"]
    assert instance.challenge_rho == result["rho"]
    assert instance.c_x == result["C_X"]
    assert instance.c_theta == result["C_theta"]
    assert instance.c_a == result["C_A"]
    assert instance.r_x == result["r_X"]
    assert instance.assign == result["assignments"].tolist()
    assert public["proof"]["ok"] is True
    assert public["proof"]["verify_ok"] is True


def test_tasklog_enforces_budget():
    log = TaskLog(Q=1)
    log.record_commitment("prover0", C_X=1, C_theta=2)
    try:
        log.record_commitment("prover0", C_X=3, C_theta=4)
        assert False, "expected budget exhaustion to raise"
    except PermissionError:
        pass


def test_protocol_uses_supplied_randomness_source():
    X, centroids = _small_instance()
    rho = bytes(range(32))
    calls = []
    result = run_protocol(
        X, centroids, m=8, q=2, scale=256,
        challenge_fn=lambda: calls.append("challenge") or rho,
    )
    assert calls == ["challenge"]
    assert result["rho"] == rho
    assert result["seed"] == derive_seed(result["C_X"], result["C_theta"], rho)
