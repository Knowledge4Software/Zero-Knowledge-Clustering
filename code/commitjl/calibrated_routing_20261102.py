"""Measured-cost batch routing. Planning is private; proof acceptance is unchanged.

The minimum below is a minimum of a fitted cost model, not a guarantee about
elapsed time. Both representations must use the same calibration procedure.
"""
from dataclasses import asdict, dataclass
import math
import time

from commitjl.cost_aware import Planner, exact_distances
from commitjl.jl_certificate_pilots import refinement_work
from experiments.pca_advantage_20261102 import endpoint_order


@dataclass(frozen=True)
class BatchCosts:
    full_seconds: float
    certificate_seconds: float
    correction_seconds: float
    full_batch_seconds: float = 0.
    certificate_batch_seconds: float = 0.
    refinement_batch_seconds: float = 0.
    margin: float = .10

    def __post_init__(self):
        values = list(asdict(self).values())
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite cost estimate')
        if min(values[:3]) <= 0 or min(values[3:6]) < 0 or not 0 <= self.margin < 1:
            raise ValueError('Invalid cost estimate or margin')


def route_cost(records, certificate, costs, extra_setup_seconds=0.):
    chosen = set(certificate)
    if any(i < 0 or i >= len(records) or not records[i]['conclusive'] for i in chosen):
        raise ValueError('A certificate route requires conclusive preflight')
    corrections = sum(records[i]['predicted_corrections'] for i in chosen)
    return (len(chosen)*costs.certificate_seconds + corrections*costs.correction_seconds
            + (len(records)-len(chosen))*costs.full_seconds
            + (costs.certificate_batch_seconds+extra_setup_seconds if chosen else 0.)
            + (costs.refinement_batch_seconds if corrections else 0.)
            + (costs.full_batch_seconds if len(chosen) < len(records) else 0.))


def choose_routes(records, costs, extra_setup_seconds=0.):
    """Minimize the two-branch cost model, including one-time key/replay costs.

With a fixed set of active branches, selecting the largest per-request savings
is optimal. Prefixes over all candidates and over zero-correction candidates
cover both refinement activation states, including avoiding the Full overhead
by certifying the complete batch. Inconclusive requests can never be selected.
"""
    if not math.isfinite(extra_setup_seconds) or extra_setup_seconds < 0:
        raise ValueError('Invalid extra setup cost')
    if any(not isinstance(r['predicted_corrections'], int) or
           not 0 <= r['predicted_corrections'] <= 16 for r in records):
        raise ValueError('Invalid correction count')
    eligible = [i for i, r in enumerate(records) if r['conclusive']]
    score = lambda i: (costs.certificate_seconds +
                       records[i]['predicted_corrections']*costs.correction_seconds,
                       records[i]['request_id'])
    full = route_cost(records, [], costs)
    best, best_cost = [], full
    for candidates in [eligible, [i for i in eligible if not records[i]['predicted_corrections']]]:
        ordered = sorted(candidates, key=score)
        for count in range(1, len(ordered)+1):
            subset = ordered[:count]
            value = route_cost(records, subset, costs, extra_setup_seconds)
            if (value, len(subset)) < (best_cost, len(best)):
                best, best_cost = subset, value
    if best_cost >= full*(1-costs.margin):
        best, best_cost = [], full
    return dict(certificate=sorted(best), full=[i for i in range(len(records)) if i not in best],
                predicted_routed_seconds=best_cost, predicted_full_seconds=full,
                predicted_savings_seconds=full-best_cost, costs=asdict(costs),
                extra_setup_seconds=extra_setup_seconds,
                guarantee='Minimum fitted cost subject to margin; runtime is not guaranteed.')


class CalibratedPlanner(Planner):
    def request(self, centers, request_id):
        started = time.perf_counter()
        bounds = self.bounds(centers)
        lo, hi = bounds['change_lower_scaled'], bounds['change_upper_scaled']
        if lo > 0 or hi < 0 or lo == hi == 0:
            order = []
            work = refinement_work(bounds, [0]*len(self.x), [], 0)
        else:
            distance = exact_distances(self.x, centers)
            exact = [(int(row.min())-int(row[g]))*self.denominator
                     for row, g in zip(distance, self.groups)]
            order = endpoint_order(bounds, exact)[:self.cap]
            work = refinement_work(bounds, exact, order, self.cap)
        return dict(request_id=request_id, conclusive=work['decision'] != 'inconclusive',
                    predicted_decision=work['decision'], predicted_corrections=work['exact_rows'],
                    refinement_indices=order[:work['exact_rows']],
                    native_seconds=time.perf_counter()-started)

    def batch(self, centers, extra_setup_seconds=0.):
        started = time.perf_counter()
        records = [self.request(c, i) for i, c in enumerate(centers)]
        plan = choose_routes(records, self.costs, extra_setup_seconds)
        for i, record in enumerate(records):
            selected = i in plan['certificate']
            record.update(path='certificate' if selected else 'full',
                          reason='measured_batch_cost' if selected else
                          'preflight_inconclusive' if not record['conclusive'] else 'measured_cost_or_margin')
            if not selected:
                record['refinement_indices'] = []
        return dict(plan, records=records,
                    native_seconds=self.preparation_seconds+time.perf_counter()-started,
                    privacy='Path choices and correction indices are public outputs; exact planning is private.')
