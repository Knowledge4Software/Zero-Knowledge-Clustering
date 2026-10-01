"""Symbolic per-stage circuit-cost model, matching Table "Symbolic circuit
cost model" in manuscript.tex (Sec. "Circuit realization and checked
arithmetic").

ASSUMPTIONS (documented; the manuscript table gives invocation counts and
qualitative arithmetic/lookup notes but not closed-form constraint-count
formulas per gate -- these are reasonable symbolic models consistent with
the manuscript's stated invocation counts, using a small number of named
per-gate constants so a user can recalibrate against measured gnark counts
in P3):
  - c_hash: constraints per in-circuit hash invocation (Poseidon/MiMC-like), default 1 (i.e. counts are reported in units of "hash invocations"; multiply by the backend's actual per-hash constraint count to get raw constraints).
  - c_range: constraints per range-check / lookup of a bounded value, default 1 (same convention: "range-check invocations").
  - c_cmp: constraints per less-than / comparison gate, default 1.
All "constraints" fields below are therefore given as *symbolic unit
counts* (invocations of the named primitive), not raw R1CS/PLONK gate
counts, consistent with "Table circuit-cost" using invocation counts as the
primary column and leaving raw constraints backend-dependent.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class CostModelParams:
    c_hash: float = 1.0     # constraints per hash invocation
    c_range: float = 1.0    # constraints per range-check invocation
    c_cmp: float = 1.0      # constraints per comparison/lookup invocation
    c_add: float = 1.0      # constraints per addition
    c_mul: float = 1.0      # constraints per multiplication/squaring


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return max(p, 1)


def commitment_authentication_cost(n: int, d: int, params: CostModelParams = CostModelParams()) -> Dict[str, float]:
    """nd leaves, padded to N = 2^ceil(log2(nd)); (N-1) hash invocations for
    the Merkle tree, plus N leaf-bound range checks."""
    nd = n * d
    N = _next_pow2(nd)
    hash_invocations = max(N - 1, 0)
    return {
        "leaves": nd, "padded_N": N,
        "hash_invocations": hash_invocations,
        "range_invocations": N,
        "constraints": hash_invocations * params.c_hash + N * params.c_range,
    }


def matrix_authentication_cost(q: int, d: int, params: CostModelParams = CostModelParams()) -> Dict[str, float]:
    """qd sparse records; one hash digest over the whole list (few hash
    invocations, modeled as O(qd) absorbed field elements via a sponge with
    rate-1 absorption, i.e. ~qd/rate hash permutation calls; we take rate=1
    as a conservative default) plus qd row/sign/distinctness checks."""
    qd = q * d
    hash_invocations = qd  # conservative: one permutation call per absorbed record
    return {
        "records": qd,
        "hash_invocations": hash_invocations,
        "range_invocations": qd,  # row < m and sign in {-1,+1} checks
        "constraints": hash_invocations * params.c_hash + qd * params.c_range,
    }


def sparse_projection_cost(n: int, q: int, d: int, params: CostModelParams = CostModelParams()) -> Dict[str, float]:
    """nqd signed contributions -> ~nqd additions, plus accumulator range checks (~nq, one per output coordinate per point... approximated as n*m via the m<=qd relation is omitted; we report per-contribution bound checks as nqd conservatively)."""
    nqd = n * q * d
    return {
        "contributions": nqd,
        "add_invocations": nqd,
        "range_invocations": nqd,
        "constraints": nqd * params.c_add + nqd * params.c_range,
    }


def projected_distance_cost(n: int, k: int, m: int, params: CostModelParams = CostModelParams()) -> Dict[str, float]:
    """nkm coordinates: nkm squarings, nk(m-1) additions, nkm difference
    bound checks."""
    nkm = n * k * m
    nk = n * k
    squarings = nkm
    additions = nk * max(m - 1, 0)
    return {
        "coordinates": nkm,
        "mul_invocations": squarings,
        "add_invocations": additions,
        "range_invocations": nkm,
        "constraints": squarings * params.c_mul + additions * params.c_add + nkm * params.c_range,
    }


def argmin_cost(n: int, k: int, params: CostModelParams = CostModelParams()) -> Dict[str, float]:
    """n(k-1) comparisons in a tournament with lowest-index tie handling."""
    comparisons = n * max(k - 1, 0)
    return {
        "comparisons": comparisons,
        "cmp_invocations": comparisons,
        "constraints": comparisons * params.c_cmp,
    }


def centroid_update_cost(T: int, k: int, m: int, params: CostModelParams = CostModelParams()) -> Dict[str, float]:
    """Lloyd-variant only (out of primary scope, symbolic for completeness):
    Tkm coordinates, division/inverse-heavy (modeled as one costly 'div'
    unit per coordinate, plus nonempty-cluster checks)."""
    Tkm = T * k * m
    return {
        "coordinates": Tkm,
        "div_invocations": Tkm,
        "nonempty_check_invocations": T * k,
        "constraints": Tkm * (params.c_mul + params.c_cmp),  # inverse ~ mul+range in many backends
    }


def full_dimensional_baseline_cost(n: int, k: int, d: int, params: CostModelParams = CostModelParams()) -> Dict[str, float]:
    """Reference point: nkd coordinates, nkd squarings, nk(d-1) additions,
    nkd difference bound checks (i.e. projected_distance_cost with m=d)."""
    return projected_distance_cost(n, k, d, params)


def total_circuit_cost(n: int, d: int, m: int, k: int, q: int,
                        params: CostModelParams = CostModelParams(),
                        include_argmin: bool = True) -> Dict[str, float]:
    """Aggregate symbolic cost for the primary (sparse-JL) circuit variant:
    commitment authentication + matrix authentication + sparse projection +
    projected distance + argmin."""
    parts = {
        "commitment_auth": commitment_authentication_cost(n, d, params),
        "matrix_auth": matrix_authentication_cost(q, d, params),
        "sparse_projection": sparse_projection_cost(n, q, d, params),
        "projected_distance": projected_distance_cost(n, k, m, params),
    }
    total = sum(p["constraints"] for p in parts.values())
    if include_argmin:
        parts["argmin"] = argmin_cost(n, k, params)
        total += parts["argmin"]["constraints"]
    parts["total_constraints"] = total
    return parts


def full_vs_sparse_reduction(n: int, d: int, m: int, k: int, q: int,
                              params: CostModelParams = CostModelParams()) -> Dict[str, float]:
    """Ratio of full-dimensional baseline circuit constraints to the primary
    sparse-JL circuit's constraints, for the reduction-percentage claims in
    RQ2 ('reduces constraints by X%')."""
    full = full_dimensional_baseline_cost(n, k, d, params)["constraints"]
    full += commitment_authentication_cost(n, d, params)["constraints"]  # both variants pay this
    full += argmin_cost(n, k, params)["constraints"]
    sparse = total_circuit_cost(n, d, m, k, q, params)["total_constraints"]
    reduction_pct = 100.0 * (1 - sparse / full) if full > 0 else float("nan")
    return {"full_constraints": full, "sparse_constraints": sparse, "reduction_pct": reduction_pct}
