import pytest
from commitjl.group_moment_reference import prepare_groups, evaluate
from commitjl.reuse_certificates import exact_reference


@pytest.mark.parametrize('groups', [[0, 0, 1, 1], [1, 0, 0, 1]])
def test_exact_aggregate_with_switches_and_arbitrary_fixed_groups(groups):
    rows = [[0], [4], [9], [10]]
    old, new = [[0], [10]], [[0], [6]]
    old_labels, old_cost = exact_reference(rows, old)
    new_labels, new_cost = exact_reference(rows, new)
    prepared = prepare_groups(rows, groups, 2)
    before = dict(labels=old_labels, certified_flags=[True]*4)
    after = dict(labels=new_labels, certified_flags=[True]*4)
    result = evaluate(rows, prepared, old, new, before, after)
    assert result['exact_change'] == new_cost-old_cost
    assert 1 in result['uncertain_indices']  # actual assignment switches
    assert all(i in result['uncertain_indices'] for i, g in enumerate(groups)
               if old_labels[i] != g or new_labels[i] != g)
    before['certified_flags'] = [False]*4
    full = evaluate(rows, prepared, old, new, before, after)
    assert full['exact_rows'] == 4
    assert full['exact_change'] == result['exact_change']


def test_large_integer_moments_without_individual_exact_rows():
    v = 10**30
    rows = [[v], [v+1]]
    old, new = [[v], [-v]], [[v+2], [-v]]
    cert = dict(labels=[0, 0], certified_flags=[True, True])
    result = evaluate(rows, prepare_groups(rows, [0, 0], 2), old, new, cert, cert)
    assert result['exact_rows'] == 0
    assert result['exact_change'] == exact_reference(rows, new)[1]-exact_reference(rows, old)[1]
