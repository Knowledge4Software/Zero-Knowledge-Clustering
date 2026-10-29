"""Native exact-integer block residual bound; no new proof relation is implied."""
import numpy as np
from .paired_certificates import change_parameters
from .reuse_certificates import integer_stats, ceil_sqrt


def residual_vectors(model, rows):
    b = np.asarray(model['basis'], dtype=np.int64)
    w = np.asarray(rows, dtype=np.int64)-np.asarray(model['centers'][0], dtype=np.int64)
    t = int(model['basis_scale'])
    dot_bound = int(np.abs(w).max(initial=0))*max(sum(abs(int(x)) for x in row) for row in b)
    bound = t*t*int(np.abs(w).max(initial=0))+dot_bound*max(sum(abs(int(x)) for x in col) for col in b.T)
    if max(dot_bound, bound) >= 2**63:
        w, b = w.astype(object), b.astype(object)
    return t*t*w-(w@b.T)@b


def block_changes(model, rows, old, new, block_counts=(1, 4, 16, 49)):
    """Return per-row change envelopes, each intersected with the global bound.

    Blocks are fixed contiguous coordinates. Ceilings use Python integers.
    The minimum of two valid cross-term upper bounds remains valid.
    """
    parameters = change_parameters(model, old, new)
    h, energy = integer_stats(model, rows)
    scores = h.astype(object)@np.asarray(parameters['coefficients'], dtype=object).T
    scores += np.asarray(parameters['constants'], dtype=object)
    r = residual_vectors(model, rows)
    ro, rn = residual_vectors(model, old), residual_vectors(model, new)
    directions = np.asarray([a-b for a in rn for b in ro])
    global_error = np.asarray([ceil_sqrt(x) for x in energy], dtype=object)[:, None]
    global_error = global_error*np.asarray(parameters['residual_delta_radii'], dtype=object)
    output = {}
    for count in block_counts:
        if not 1 <= count <= r.shape[1]:
            raise ValueError('invalid block count')
        error = np.zeros(global_error.shape, dtype=object)
        for indices in np.array_split(np.arange(r.shape[1]), count):
            row_radius = [ceil_sqrt(sum(int(v)**2 for v in row)) for row in r[:, indices]]
            center_radius = [ceil_sqrt(sum(int(v)**2 for v in row)) for row in directions[:, indices]]
            error += np.asarray(row_radius, dtype=object)[:, None]*np.asarray(center_radius, dtype=object)
        error = np.minimum(error, global_error)*2*int(model['basis_scale'])**2
        shape = (len(rows), len(new), len(old))
        lower = (scores-error).reshape(shape).max(axis=2).min(axis=1)
        upper = (scores+error).reshape(shape).min(axis=1).max(axis=1)
        assert all(a <= b for a, b in zip(lower, upper))
        output[count] = dict(lower=lower, upper=upper, denominator=parameters['cost_denominator'])
    return output
