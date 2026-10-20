import numpy as np

from experiments.server_claim_check import native_controls


def test_exact_statistics_detect_a_known_kernel_cost_change_without_a_jl_challenge():
    rng=np.random.default_rng(984)
    x=rng.normal(size=(24,32));centers=rng.normal(size=(5,32))
    result=native_controls(x,centers,'fixture',5)
    assert result['exact_integer_affine_distances_equal']
    assert result['float_head_energy_max_relative_error']<1e-12
    assert abs(result['known_linear_map_claim_inflation']-1)<1e-10
    assert result['true_cost_inflation']>5
    assert abs(result['known_no_jl_head_energy_inflation']-result['true_cost_inflation'])<1e-10
