"""Routing decisions compared with an independent exhaustive subset oracle."""
import itertools
import json
import math
import random

import numpy as np
import pytest

from commitjl.calibrated_routing_20261102 import BatchCosts, CalibratedPlanner, choose_routes, route_cost
from experiments.pca_advantage_20261102 import SCALE, model_for
from analysis.server_routing_calibration_20261102 import calibrate_case


def test_optimizer_matches_exhaustive_subsets_including_branch_overheads():
    rng = random.Random(907)
    for _ in range(80):
        n = rng.randrange(1, 9)
        records = [dict(request_id=i, conclusive=rng.random()>.2,
                        predicted_corrections=rng.randrange(17)) for i in range(n)]
        c = BatchCosts(rng.uniform(1, 150), rng.uniform(1, 60), rng.uniform(.1, 8),
                       rng.uniform(0, 150), rng.uniform(0, 100), rng.uniform(0, 100), margin=0)
        setup = rng.uniform(0, 100)
        eligible = [i for i, r in enumerate(records) if r['conclusive']]
        exact = min(route_cost(records, subset, c, setup)
                    for count in range(len(eligible)+1)
                    for subset in itertools.combinations(eligible, count))
        selected = choose_routes(records, c, setup)
        assert selected['predicted_routed_seconds'] == pytest.approx(exact)
        assert all(records[i]['conclusive'] for i in selected['certificate'])


def test_margin_and_setup_can_prevent_activation_without_attempting_corrections():
    rows = [dict(request_id=i, conclusive=True, predicted_corrections=0) for i in range(4)]
    assert choose_routes(rows, BatchCosts(100, 20, 2))['certificate'] == [0, 1, 2, 3]
    assert choose_routes(rows, BatchCosts(100, 20, 2), 400)['certificate'] == []
    assert choose_routes(rows, BatchCosts(100, 95, 2))['certificate'] == []


def test_avoiding_full_fixed_cost_can_justify_one_costly_certificate():
    rows = [dict(request_id=0, conclusive=True, predicted_corrections=0),
            dict(request_id=1, conclusive=True, predicted_corrections=16)]
    assert choose_routes(rows, BatchCosts(20, 5, 2, full_batch_seconds=100, margin=0))['certificate'] == [0, 1]


def test_inconclusive_update_always_uses_full_and_invalid_inputs_fail():
    rows = [dict(request_id=0, conclusive=False, predicted_corrections=16)]
    assert choose_routes(rows, BatchCosts(100, 1, 1))['full'] == [0]
    with pytest.raises(ValueError):
        route_cost(rows, [0], BatchCosts(100, 1, 1))
    with pytest.raises(ValueError):
        BatchCosts(math.nan, 1, 1)
    with pytest.raises(ValueError):
        BatchCosts(100, 1, 1, margin=1)
    with pytest.raises(ValueError):
        choose_routes(rows, BatchCosts(100, 1, 1), -1)


def test_preflight_handles_equal_updates_and_keeps_exact_correction_budget():
    old = np.array([[-10, 0, 0], [10, 0, 0]], dtype=np.int64)
    rows = np.array([[-9, 1, 0], [-8, 0, 0], [8, 0, 0], [9, 1, 0]], dtype=np.int64)
    basis = np.array([[SCALE, 0, 0]], dtype=np.int64)
    p = CalibratedPlanner(model_for(basis, old, 4, 1.), rows, BatchCosts(100, 1, 1))
    result = p.batch([old.tolist(), (old+np.array([0, 1, 0])).tolist()])
    assert result['records'][0]['predicted_decision'] == 'equal'
    assert sorted(result['certificate']+result['full']) == [0, 1]
    assert all(r['predicted_corrections'] <= 16 for r in result['records'])


def test_calibration_excludes_cached_old_full_proofs_and_charges_key_loading(tmp_path):
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    name, method = 'toy', 'pca'
    base = tmp_path/'matched'/name/(method+'_base')
    ref = tmp_path/'matched'/name/(method+'_refined')
    warm = tmp_path/'warm'/name/(name+'_ordered_head_warm01')
    def samples(seconds, sets):
        return dict(complete=True, samples=[dict(centre_set=s, prove_seconds=[t]) for s, t in zip(sets, seconds)])
    put(base/'input.json', dict(reuse_centers=[[], [], []], model=dict(n=4, d=3, basis=[[1, 0, 0]], input_bits=12)))
    put(tmp_path/'matched'/name/method/'full/input.json', dict(queries=[[0, 1, 2, 3]]))
    put(warm/'baseline_full/private/full_reuse_8.json', samples([1000, 20, 20], [0, 1, 2]))
    put(warm/'checked.json', dict(requests=2, resources=dict(baseline_full_prove=dict(wall_seconds=50),
                                                        baseline_verify=dict(wall_seconds=2))))
    put(base/'bundle/group_query.json', samples([4, 4], [0, 1]))
    put(ref/'bundle/refinement_prove.json', samples([2, 2, 2], [0, 1, 1]))
    put(base/'status.json', dict(resources=dict(group_query=dict(wall_seconds=13), group_verify=dict(wall_seconds=1))))
    put(ref/'status.json', dict(resources=dict(refinement_prove=dict(wall_seconds=9), refinement_verify=dict(wall_seconds=3))))
    costs, pin = calibrate_case(tmp_path, name, method)
    assert costs.full_seconds == 20
    assert costs.full_batch_seconds == 12
    assert costs.certificate_seconds == 4
    assert costs.certificate_batch_seconds == 6
    assert costs.correction_seconds == 2
    assert costs.refinement_batch_seconds == 5
    assert len(pin['files']) == 8
