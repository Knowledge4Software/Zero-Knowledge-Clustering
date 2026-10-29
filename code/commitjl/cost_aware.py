"""Untrusted, cost-aware witness planning for the existing compact relation.

Planning uses the prover's original rows. It does not certify an answer: every
selected compact/refinement or Full proof must still pass public verification.
The public path and refinement indices are additional observable outputs.
"""
from dataclasses import asdict, dataclass
import math
import time

import numpy as np

from .group_change_bounds import advantage_bounds, assigned_change
from .group_moment_reference import prepare_groups
from .jl_certificate_pilots import hint_order, refinement_work
from .reuse_certificates import integer_stats


def exact_distances(rows, centers):
    """Integer distances, with an overflow-safe fallback for larger inputs."""
    x = np.asarray(rows, dtype=np.int64)
    c = np.asarray(centers, dtype=np.int64)
    if x.ndim != 2 or c.ndim != 2 or not len(x) or len(c) < 2 or x.shape[1] != c.shape[1]:
        raise ValueError('invalid row/center shape')
    maximum = max(abs(int(v)) for v in [x.min(), x.max(), c.min(), c.max()])
    if 4 * x.shape[1] * maximum ** 2 >= 2 ** 63:
        return np.asarray([[sum((int(a)-int(b))**2 for a, b in zip(row, center))
                            for center in c] for row in x], dtype=object)
    return ((x[:, None, :] - c[None, :, :]) ** 2).sum(2)


@dataclass(frozen=True)
class ProofCosts:
    full_seconds: float
    certificate_seconds: float
    correction_seconds: float
    refinement_replay_seconds: float = 0.
    margin: float = .1
    certificate_batch_seconds: float = 0.
    refinement_batch_seconds: float = 0.

    def __post_init__(self):
        values = [self.full_seconds, self.certificate_seconds, self.correction_seconds,
                  self.refinement_replay_seconds, self.margin,
                  self.certificate_batch_seconds, self.refinement_batch_seconds]
        if not all(math.isfinite(v) for v in values) or min(values[:3]) <= 0:
            raise ValueError('cost estimates must be finite and positive')
        if self.refinement_replay_seconds < 0 or not 0 <= self.margin < 1:
            raise ValueError('invalid replay cost or safety margin')
        if min(self.certificate_batch_seconds, self.refinement_batch_seconds) < 0:
            raise ValueError('invalid batch overhead')


class Planner:
    """Cache fixed-catalog geometry once; never generate a proof to pick a path."""
    def __init__(self, model, rows, costs, max_corrections=16):
        if not isinstance(max_corrections, int) or not 0 <= max_corrections <= 16:
            raise ValueError('correction budget must be between zero and sixteen')
        started = time.perf_counter()
        self.model = dict(model)
        self.x = np.array(rows, dtype=np.int64, copy=True)
        self.old = np.array(model['centers'], dtype=np.int64, copy=True)
        if self.x.shape != (model['n'], model['d']) or len(self.old) != model['k']:
            raise ValueError('catalog/model shape mismatch')
        self.costs, self.cap = costs, max_corrections
        self.groups = exact_distances(self.x, self.old).argmin(1)
        self.prepared = prepare_groups(self.x, self.groups.tolist(), len(self.old))
        self.heads, self.energies = integer_stats(self.model, self.x)
        self.denominator = int(model['basis_scale']) ** 6
        self.preparation_seconds = time.perf_counter() - started

    def bounds(self, centers):
        if np.asarray(centers).shape != self.old.shape:
            raise ValueError('candidate shape differs from the fixed model')
        lower, upper = advantage_bounds(self.model, self.heads, self.energies, centers, self.groups)
        assigned = assigned_change(self.prepared, self.old, centers) * self.denominator
        return dict(assigned_scaled=assigned, row_lower=lower, row_upper=upper,
                    cost_denominator=self.denominator,
                    change_lower_scaled=assigned+sum(map(int, lower)),
                    change_upper_scaled=assigned+sum(map(int, upper)))

    def request(self, centers, request_id):
        started = time.perf_counter()
        bounds = self.bounds(centers)
        order = hint_order(bounds, None, 'width')[:self.cap]
        lo, hi = bounds['change_lower_scaled'], bounds['change_upper_scaled']
        # Exact corrections are private witness planning, never verifier evidence.
        # Do not recompute them for an already conclusive base interval.
        if lo > 0 or hi < 0 or lo == hi == 0:
            work = refinement_work(bounds, [0] * len(self.x), [], 0)
        else:
            d = exact_distances(self.x, centers)
            exact = [(int(row.min())-int(row[g])) * self.denominator
                     for row, g in zip(d, self.groups)]
            work = refinement_work(bounds, exact, order, self.cap)
        estimate = self.costs.certificate_seconds + work['exact_rows'] * self.costs.correction_seconds
        if work['exact_rows']:
            estimate += self.costs.refinement_replay_seconds
        conclusive = work['decision'] != 'inconclusive'
        use_certificate = conclusive and estimate < self.costs.full_seconds * (1-self.costs.margin)
        return dict(request_id=request_id, path='certificate' if use_certificate else 'full',
                    reason=('predicted_conclusive_and_cheaper' if use_certificate else
                            'predicted_inconclusive' if not conclusive else 'estimated_proof_cost'),
                    refinement_indices=order[:work['exact_rows']] if use_certificate else [],
                    predicted_corrections=work['exact_rows'], predicted_decision=work['decision'],
                    estimated_certificate_seconds=estimate, estimated_full_seconds=self.costs.full_seconds,
                    native_seconds=time.perf_counter()-started)

    def batch(self, centers, extra_setup_seconds=0.):
        """Optional cold-batch activation; no promise beyond the cost estimates.

        With existing authenticated summaries use zero. Otherwise charge their
        extra fixed setup before deciding whether to activate the compact path.
        All candidates are retained even when the entire batch uses Full.
        """
        if not math.isfinite(extra_setup_seconds) or extra_setup_seconds < 0:
            raise ValueError('invalid extra setup cost')
        records = [self.request(c, i) for i, c in enumerate(centers)]
        savings = sum(r['estimated_full_seconds']-r['estimated_certificate_seconds']
                      for r in records if r['path'] == 'certificate')
        branch_overhead = self.costs.certificate_batch_seconds if any(r['path']=='certificate' for r in records) else 0.
        if any(r['path']=='certificate' and r['refinement_indices'] for r in records):
            branch_overhead += self.costs.refinement_batch_seconds
        # A few cheap queries must cover loading/decoding their proving keys,
        # not only their individual proof work. Keep a batch-level margin too.
        required_savings = extra_setup_seconds+branch_overhead+self.costs.margin*sum(r['estimated_full_seconds'] for r in records)
        if savings <= required_savings:
            for r in records:
                if r['path'] == 'certificate':
                    r.update(path='full', reason='batch_setup_not_amortized' if extra_setup_seconds else 'batch_overhead_or_margin', refinement_indices=[])
        return dict(records=records, certificate=[r['request_id'] for r in records if r['path']=='certificate'],
                    full=[r['request_id'] for r in records if r['path']=='full'],
                    native_seconds=self.preparation_seconds+sum(r['native_seconds'] for r in records),
                    costs=asdict(self.costs), extra_setup_seconds=extra_setup_seconds,
                    predicted_branch_overhead_seconds=branch_overhead,
                    required_gross_savings_seconds=required_savings,
                    predicted_gross_savings_seconds=savings,
                    privacy='Public path and selected refinement indices are observable. Private exact corrections are only planning hints.')
