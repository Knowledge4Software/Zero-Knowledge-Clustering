import numpy as np
import pytest
from commitjl.reuse_certificates import exact_squared_norms,integer_stats


@pytest.mark.parametrize('width',[1,2,79,784,4096])
def test_limb_energy_matches_unbounded_integer_oracle(width):
    rng=np.random.default_rng(173)
    rows=rng.integers(-(2**62),2**62,size=(7,width),dtype=np.int64)
    rows[0,0]=-(2**63);rows[1,0]=2**63-1
    oracle=[sum(int(v)**2 for v in row) for row in rows]
    assert exact_squared_norms(rows).tolist()==oracle
    assert exact_squared_norms(rows.astype(object)).tolist()==oracle


def test_rounded_statistics_preserve_heads_and_energy():
    rng=np.random.default_rng(191)
    rows=rng.integers(-4096,4097,size=(513,87),dtype=np.int64)
    basis=rng.integers(-16384,16385,size=(13,87),dtype=np.int64)
    center=rng.integers(-1000,1000,size=87,dtype=np.int64)
    model=dict(basis=basis.tolist(),basis_scale=16384,centers=[center.tolist()])
    heads,energy=integer_stats(model,rows)
    w=rows.astype(object)-center.astype(object);b=basis.astype(object)
    dot=w@b.T;residual=16384**2*w-dot@b
    assert np.array_equal(heads,16384*dot)
    assert energy.tolist()==[sum(int(v)**2 for v in row) for row in residual]
