"""Commitment scheme: C_X (Merkle root over field-encoded signed leaves,
padded to a power of two, with blinding) and C_theta / C_A (Poseidon/MiMC
sponge-style digests over encoded fields).

Manuscript reference: Alg. 1 step 2 "(C_X, r_X) <- Commit(Encode(X, mu))";
Table circuit-cost row "Commitment authentication ... nd leaves, padded to N".
We use the MiMC-BN254 hash from hashing.py as the tree/sponge primitive
(a documented stand-in for the paper's placeholder \\placeholder{commitment
hash}; Poseidon and MiMC play the same structural role here).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .encoding import encode_theta, signed_to_field, dense_to_sparse_list, encode_sparse_list
from .hashing import BN254_FR, mimc_hash, mimc_hash_bytes


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return max(p, 1)


def _leaf_hash(value_field: int, salt: int, p: int = BN254_FR) -> int:
    return mimc_hash([value_field, salt], p)


def _node_hash(left: int, right: int, p: int = BN254_FR) -> int:
    return mimc_hash([left, right], p)


@dataclass
class MerkleCommitment:
    root: int
    leaves: List[int]        # field-encoded (pre-hash) leaf values, padded length N
    leaf_hashes: List[int]   # hashed leaves, padded length N
    salt: int                # blinding r_X
    n_real: int               # number of real (non-padding) leaves


def commit_matrix(X: np.ndarray, p: int = BN254_FR, salt: int | None = None) -> MerkleCommitment:
    """C_X = Poseidon/MiMC-Merkle root over n*d field-encoded signed leaves.

    Leaves are the row-major flattening of X, mapped via the canonical
    signed->field encoding (negative x -> p + x), then hashed together with a
    single blinding salt r_X (added into every leaf hash) and padded to the
    next power of two using zero leaves.
    """
    X = np.asarray(X)
    n, d = X.shape
    nd = n * d
    if salt is None:
        salt = int.from_bytes(os.urandom(32), "big") % p

    flat = X.astype(np.int64).ravel(order="C")
    leaves = [signed_to_field(int(v), p) for v in flat]

    N = _next_pow2(max(nd, 1))
    leaves_padded = leaves + [0] * (N - nd)

    leaf_hashes = [_leaf_hash(v, salt, p) for v in leaves_padded]

    level = leaf_hashes
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            nxt.append(_node_hash(level[i], level[i + 1], p))
        level = nxt
    root = level[0]

    return MerkleCommitment(root=root, leaves=leaves_padded, leaf_hashes=leaf_hashes, salt=salt, n_real=nd)


def merkle_proof(commitment: MerkleCommitment, index: int, p: int = BN254_FR) -> List[Tuple[int, str]]:
    """Return the authentication path (sibling_hash, 'L'/'R') for leaf `index`."""
    level = list(commitment.leaf_hashes)
    path = []
    idx = index
    while len(level) > 1:
        sib_idx = idx ^ 1
        side = "L" if sib_idx < idx else "R"
        path.append((level[sib_idx], side))
        nxt = []
        for i in range(0, len(level), 2):
            nxt.append(_node_hash(level[i], level[i + 1], p))
        level = nxt
        idx //= 2
    return path


def verify_merkle_proof(leaf_value: int, salt: int, index: int, path: List[Tuple[int, str]],
                          root: int, p: int = BN254_FR) -> bool:
    h = _leaf_hash(leaf_value, salt, p)
    for sib, side in path:
        if side == "L":
            h = _node_hash(sib, h, p)
        else:
            h = _node_hash(h, sib, p)
    return h == root


def commit_theta(centroids: np.ndarray, m: int, q: int, mu: bytes = b"", p: int = BN254_FR) -> int:
    """C_theta = H_theta(Encode(theta, mu)) -- MiMC digest over the canonical
    byte encoding of (centroids, m, q, mu)."""
    encoded = encode_theta(centroids, m, q, mu)
    return mimc_hash_bytes(encoded, p)


def commit_sparse_matrix(A_dense: np.ndarray, p: int = BN254_FR) -> int:
    """C_A = H_A(Encode(A)) -- MiMC digest over the canonical sparse-list
    byte encoding of A."""
    records = dense_to_sparse_list(A_dense)
    encoded = encode_sparse_list(records)
    return mimc_hash_bytes(encoded, p)
