"""Scientific invariants for the isolated basis/rank server study."""
import itertools
from pathlib import Path

import numpy as np
import pytest

from experiments.pca_advantage_20261102 import (
    SCALE, endpoint_order, evaluate, fit_families, proposal_banks,
    select_validation, selection_score)
from commitjl.jl_certificate_pilots import hint_order, refinement_work
from commitjl.protected_jl import centroid_basis


def test_gain_order_minimizes_strict_endpoint_correction_count():
    for assigned, exact in [(14, [-8, -2, -1, 0]), (0, [-8, -2, -1, 0])]:
        bounds = dict(assigned_scaled=assigned, row_lower=[-10, -10, -10, -10],
                      row_upper=[0, 0, 0, 0], change_lower_scaled=assigned-40,
                      change_upper_scaled=assigned, cost_denominator=1)
        selected = refinement_work(bounds, exact, endpoint_order(bounds, exact), 4)
        minimum = 5
        for count in range(5):
            for subset in itertools.combinations(range(4), count):
                w = refinement_work(bounds, exact, list(subset), count)
                if w['decision'] != 'inconclusive':
                    minimum = min(minimum, w['exact_rows'])
        assert selected['exact_rows'] == minimum


def test_gain_order_rejects_invalid_corrections_and_handles_equality():
    b = dict(assigned_scaled=2, row_lower=[-4, -1], row_upper=[0, 0],
             change_lower_scaled=-3, change_upper_scaled=2, cost_denominator=1)
    with pytest.raises(ValueError):
        endpoint_order(b, [-5, 0])
    assert endpoint_order(b, [-2, 0]) == hint_order(b, None, 'width')


def test_history_bases_preserve_old_span_and_training_orthogonality():
    rng = np.random.default_rng(33)
    past = rng.integers(-20, 21, size=(96, 12), dtype=np.int64)
    old = past[[0, 32, 64]]
    bases, seconds = fit_families(past, old, 97, max_rank=8)
    protected = centroid_basis(old)
    for name in ['centroid', 'history25', 'history75']:
        np.testing.assert_array_equal(bases[name][:protected.shape[1]],
                                      np.rint(protected.T*SCALE).astype(np.int64))
    assert set(bases) == {'pca', 'centroid', 'history25', 'history75'}
    assert all(v.shape == (8, 12) and np.max(np.abs(v)) <= SCALE for v in bases.values())
    assert all(v >= 0 for v in seconds.values())


def test_native_candidate_intervals_cover_truth_for_all_bases_and_ranks():
    rng = np.random.default_rng(35)
    past = rng.integers(-12, 13, size=(96, 12), dtype=np.int64)
    old = past[[0, 32, 64]]
    future = rng.integers(-12, 13, size=(96, 12), dtype=np.int64)
    rows = rng.integers(-12, 13, size=(16, 12), dtype=np.int64)
    candidates, _ = proposal_banks(future, old, count=4)
    bases, _ = fit_families(past, old, 99, max_rank=8)
    results = []
    for name, basis in bases.items():
        for rank in [2, 4, 8]:
            results.extend(evaluate(basis, old, rows, candidates, 1., name, rank))
    assert len(results) == 48
    assert all(int(r['lower']) <= int(r['truth']) <= int(r['upper']) for r in results)
    for r in results:
        if r['width_order_conclusive']:
            assert r['conclusive'] and r['corrections'] <= r['width_order_corrections']


def test_both_methods_receive_same_selection_formula_and_test_fields_unused():
    records = []
    for family in ['pca', 'centroid', 'history25', 'history75']:
        for rank in [4, 8]:
            for request in range(4):
                records.append(dict(family=family, rank=rank, request=request,
                                    conclusive=rank == 8, corrections=2))
    a = select_validation(records, 128, 784, 5)
    swapped = [dict(r, family='pca' if r['family'] == 'centroid' else
                    'centroid' if r['family'] == 'pca' else r['family'],
                    test_outcome='arbitrary ignored field') for r in records]
    b = select_validation(swapped, 128, 784, 5)
    assert a['pca']['score'] == b['pca']['score']
    assert a['ours']['score'] == b['ours']['score']
    assert selection_score(records[:4], 128, 784, 5) == selection_score(
        [dict(r, family='anything') for r in records[:4]], 128, 784, 5)


def test_proposals_are_distinct_training_only_and_reproducible():
    rng = np.random.default_rng(37)
    old = np.array([[-10, 0, 0], [10, 0, 0]], dtype=np.int64)
    incoming = rng.integers(-15, 16, size=(256, 3), dtype=np.int64)
    a, recipe = proposal_banks(incoming, old)
    b, other_recipe = proposal_banks(incoming.copy(), old)
    np.testing.assert_array_equal(a, b)
    assert recipe == other_recipe
    assert len({row.tobytes() for row in a}) == 8
    assert not any(np.array_equal(row, old) for row in a)


def test_refinement_copy_changes_only_witness_order_and_keeps_proof_checks():
    base = Path('code/campaigns/server_all_sources_refinement_20261101.py').read_text()
    new = Path('code/campaigns/server_pca_advantage_refinement_20261102.py').read_text()
    expected = base.replace(
        '# Versioned all-source coordinator: compiler resource ceiling only; proof relations unchanged.',
        '# New November2 experiment: endpoint-gain witness planning; proof acceptance relations unchanged.', 1)
    expected = expected.replace('from commitjl.jl_certificate_pilots import hint_order',
                                'from experiments.pca_advantage_20261102 import endpoint_order', 1)
    expected = expected.replace(
        "ordering='Descending canonical row interval width, index tie breaks; stop on first conclusive interval.'",
        "ordering='Prover-private exact endpoint gain, index tie breaks; planning charged; stop on first conclusive interval.'", 1)
    expected = expected.replace("                    orders[index]=hint_order(b,None,'width')[:16]",
        "                    distance=((x[:,None,:]-np.asarray(centers,dtype=np.int64)[None,:,:])**2).sum(2)\n"
        "                    exact=[(int(row.min())-int(row[g]))*b['cost_denominator'] for row,g in zip(distance,groups)]\n"
        "                    orders[index]=endpoint_order(b,exact)[:16]", 1)
    assert new == expected
