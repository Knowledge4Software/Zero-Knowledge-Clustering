import numpy as np
from experiments.rank79_independent_replication import trace
from experiments.review_update_decisions import lloyd_snapshots


def test_fresh_trace_matches_independent_existing_float_lloyd_implementation():
    x=np.random.default_rng(713).integers(-50,51,(101,17))
    expected=lloyd_snapshots(x,5,113)
    actual=trace(x,5,113)
    assert actual.keys()==expected.keys()
    for step in actual:np.testing.assert_array_equal(actual[step],expected[step])
