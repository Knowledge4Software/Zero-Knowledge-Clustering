import numpy as np

from commitjl.commit import commit_matrix, merkle_proof, verify_merkle_proof, commit_theta, commit_sparse_matrix
from commitjl.encoding import encode_matrix_X, signed_to_field, field_to_signed
from commitjl.sparse_gen import sparse_matrix_gen


def test_signed_field_roundtrip():
    p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
    for x in [-100, -1, 0, 1, 100, 12345]:
        f = signed_to_field(x, p)
        back = field_to_signed(f, p, bound=100000)
        assert back == x


def test_commit_matrix_deterministic_with_fixed_salt():
    X = np.array([[1, -2, 3], [4, 5, -6]])
    c1 = commit_matrix(X, salt=42)
    c2 = commit_matrix(X, salt=42)
    assert c1.root == c2.root


def test_commit_matrix_changes_with_data():
    X1 = np.array([[1, -2, 3], [4, 5, -6]])
    X2 = np.array([[1, -2, 3], [4, 5, -7]])
    c1 = commit_matrix(X1, salt=1)
    c2 = commit_matrix(X2, salt=1)
    assert c1.root != c2.root


def test_merkle_proof_verifies():
    X = np.array([[1, -2, 3, 4], [5, -6, 7, 8], [9, 10, -11, 12]])
    commitment = commit_matrix(X, salt=7)
    for idx in range(commitment.n_real):
        path = merkle_proof(commitment, idx)
        leaf_val = commitment.leaves[idx]
        assert verify_merkle_proof(leaf_val, commitment.salt, idx, path, commitment.root)


def test_merkle_proof_rejects_tampered_leaf():
    X = np.array([[1, 2], [3, 4]])
    commitment = commit_matrix(X, salt=7)
    path = merkle_proof(commitment, 0)
    tampered_leaf = commitment.leaves[0] + 1
    assert not verify_merkle_proof(tampered_leaf, commitment.salt, 0, path, commitment.root)


def test_commit_theta_and_C_A_deterministic():
    centroids = np.array([[1, 2], [3, 4]])
    c1 = commit_theta(centroids, m=8, q=2)
    c2 = commit_theta(centroids, m=8, q=2)
    assert c1 == c2

    _, A = sparse_matrix_gen(b"ca-seed", m=8, d=4, q=2)
    ca1 = commit_sparse_matrix(A)
    ca2 = commit_sparse_matrix(A)
    assert ca1 == ca2


def test_encode_matrix_X_deterministic():
    X = np.array([[1, -2], [3, 4]])
    b1 = encode_matrix_X(X, scale=256)
    b2 = encode_matrix_X(X, scale=256)
    assert b1 == b2
    assert b1.startswith(b"CommitJL-v1")
