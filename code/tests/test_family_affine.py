"""Independent original-distance oracle for known-family numerical scores."""
import pytest
from commitjl.family_affine import prepare, query


@pytest.mark.parametrize('scale', [1, 10**30])
def test_family_scores_match_original_distances_and_first_ties(scale):
    rows = [[0, 0], [scale, -scale], [-3*scale, 2*scale]]
    families = [[[scale, 0], [-scale, 0]], [[0, scale], [scale, 0], [0, scale]]]
    state = prepare(rows, families)
    for centers in families:
        actual = query(state, centers)
        distances = [[sum((x-y)**2 for x, y in zip(row, c)) for c in centers] for row in rows]
        assert actual['costs'] == [min(d) for d in distances]
        assert actual['labels'] == [d.index(min(d)) for d in distances]
        assert actual['total'] == sum(map(min, distances))


def test_new_center_requires_new_authenticated_information():
    state = prepare([[0, 1]], [[[1, 0], [-1, 0]]])
    with pytest.raises(ValueError, match='preprocessed family'):
        query(state, [[1, 0], [0, 1]])


def test_inconsistent_dimensions_rejected():
    with pytest.raises(ValueError, match='dimension mismatch'):
        prepare([[1, 2]], [[[1], [2]]])
