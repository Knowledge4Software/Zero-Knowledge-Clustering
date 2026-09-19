import numpy as np

from commitjl.protocol import run_protocol
from commitjl.randomness import TaskLog


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
    from commitjl.commit import commit_matrix, commit_theta, commit_sparse_matrix
    from commitjl.protocol import derive_seed
    from commitjl.sparse_gen import sparse_matrix_gen
    from commitjl.clustering import nearest_centroid_assign, psi_hat

    commitment = commit_matrix(X, salt=12345)
    C_X = commitment.root
    C_theta = commit_theta(centroids, m=8, q=2)
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


def test_tasklog_enforces_budget():
    log = TaskLog(Q=1)
    log.record_commitment("prover0", C_X=1, C_theta=2)
    try:
        log.record_commitment("prover0", C_X=3, C_theta=4)
        assert False, "expected budget exhaustion to raise"
    except PermissionError:
        pass
