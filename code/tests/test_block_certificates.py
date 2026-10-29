import numpy as np
from commitjl.block_certificates import block_changes
from commitjl.paired_certificates import paired_change
from commitjl.reuse_certificates import exact_reference


def test_block_bounds_cover_integer_truth_and_dominate_global():
    rng = np.random.default_rng(1023)
    # Deliberately nonorthogonal rounded basis; test exact Gram correction too.
    model = dict(basis=[[7, 2, 0, 1, -1, 0, 0], [0, 1, 8, 0, 1, -1, 0]],
                 basis_scale=8, centers=[[1, -2, 0, 1, 0, 0, 1], [0]*7], input_bits=12)
    rows = rng.integers(-12, 13, size=(19, 7)).tolist()
    old = rng.integers(-9, 10, size=(3, 7)).tolist()
    new = rng.integers(-9, 10, size=(2, 7)).tolist()
    bounds = block_changes(model, rows, old, new, [1, 3, 7])
    existing = paired_change(model, rows, old, new, row_outputs=True)
    assert bounds[1]['lower'].tolist() == existing['row_lower']
    assert bounds[1]['upper'].tolist() == existing['row_upper']
    for count, result in bounds.items():
        for i, row in enumerate(rows):
            truth = (exact_reference([row], new)[1]-exact_reference([row], old)[1])*result['denominator']
            assert result['lower'][i] <= truth <= result['upper'][i]
            assert result['lower'][i] >= bounds[1]['lower'][i]
            assert result['upper'][i] <= bounds[1]['upper'][i]


def test_refinement_narrows_and_finishes_at_exact_change():
    from experiments.pilot_certificate_refinement import refinement
    lower = np.array([-8, -9, 1], dtype=object)
    upper = np.array([10, 12, 5], dtype=object)
    exact = np.array([3, -2, 2], dtype=object)
    for order in [[0, 1, 2], [2, 1, 0]]:
        path = refinement(lower, upper, exact, order)
        assert path[-1] == (3, 3)
        assert all(a <= 3 <= b for a, b in path)
        assert all(a <= c <= d <= b for (a, b), (c, d) in zip(path, path[1:]))
