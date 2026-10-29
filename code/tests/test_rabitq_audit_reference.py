"""Check the geometry of the estimator independently of its statistical rate."""
import numpy as np
import pytest

from commitjl.published_audit_baselines import distances
from commitjl.rabitq_audit_reference import RaBitQAudit


def test_old_anchor_query_and_zero_residual_are_exact():
    x=np.array([[0,0,0],[1,2,3],[10,10,10]])
    old=np.array([[0,0,0],[10,10,10]])
    audit=RaBitQAudit(x,old,57,100,padded_dimension=8)
    result=audit.query(old,deterministic_component_bound=True)
    exact=distances(x,old)
    for i,g in enumerate(audit.groups):
        assert result['distance_lower'][i,g]==pytest.approx(exact[i,g])
        assert result['distance_upper'][i,g]==pytest.approx(exact[i,g])
    assert np.isfinite(result['distance_lower']).all()


def test_cauchy_schwarz_envelope_contains_every_distance_without_probability():
    rng=np.random.default_rng(894)
    x=rng.integers(-15,16,(40,9));old=rng.integers(-15,16,(4,9))
    audit=RaBitQAudit(x,old,581,4000,padded_dimension=16)
    for _ in range(8):
        centers=rng.integers(-30,31,(4,9))
        result=audit.query(centers,deterministic_component_bound=True)
        exact=distances(x,centers)
        assert np.all(result['distance_lower']<=exact+1e-8)
        assert np.all(result['distance_upper']>=exact-1e-8)


def test_union_bound_tightens_only_with_more_failure_budget():
    x=np.arange(24).reshape(8,3);old=np.array([[0,0,0],[15,15,15]])
    a=RaBitQAudit(x,old,71,100,.01,padded_dimension=8)
    b=RaBitQAudit(x,old,71,1000,.01,padded_dimension=8)
    c=RaBitQAudit(x,old,71,100,.1,padded_dimension=8)
    assert 0<c.component_bound<a.component_bound<b.component_bound<1
    np.testing.assert_allclose(a.rotation@a.rotation.T,np.eye(3),atol=1e-12)
