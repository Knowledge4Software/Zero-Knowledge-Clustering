"""Native exact aggregate change plus uncertain-row corrections.

Group moments, memberships and assignment certificates require authentication
in a future proof protocol. This module does not supply that protocol.
"""
import numpy as np


def prepare_groups(rows, groups, k):
    if len(rows) != len(groups) or not len(rows) or k < 2:
        raise ValueError('invalid group catalog')
    dim = len(rows[0])
    sums = np.zeros((k, dim), dtype=object)
    counts = [0]*k
    for row, group in zip(rows, groups):
        group = int(group)
        if len(row) != dim or not 0 <= group < k:
            raise ValueError('invalid row/group')
        counts[group] += 1
        for j, value in enumerate(row):
            sums[group, j] += int(value)
    return dict(sums=sums, counts=counts, groups=list(map(int, groups)))


def evaluate(rows, prepared, old, new, old_certificate, new_certificate):
    k = len(old)
    if len(new) != k or len(prepared['counts']) != k or len(prepared['groups']) != len(rows):
        raise ValueError('fixed k and complete catalog required')
    old, new = np.asarray(old, dtype=object), np.asarray(new, dtype=object)
    norms = [sum(int(v)**2 for v in new[g])-sum(int(v)**2 for v in old[g]) for g in range(k)]
    assigned_total = sum(2*sum(int(x)*int(d) for x, d in zip(prepared['sums'][g], old[g]-new[g]))
                         + prepared['counts'][g]*norms[g] for g in range(k))
    uncertain, correction = [], 0
    for i, (row, group) in enumerate(zip(rows, prepared['groups'])):
        stable = (old_certificate['certified_flags'][i] and new_certificate['certified_flags'][i]
                  and old_certificate['labels'][i] == group and new_certificate['labels'][i] == group)
        if stable:
            continue
        uncertain.append(i)
        assigned = 2*sum(int(x)*int(d) for x, d in zip(row, old[group]-new[group]))+norms[group]
        old_min = min(sum((int(x)-int(c))**2 for x, c in zip(row, center)) for center in old)
        new_min = min(sum((int(x)-int(c))**2 for x, c in zip(row, center)) for center in new)
        correction += new_min-old_min-assigned
    return dict(exact_change=assigned_total+correction, assigned_total=assigned_total,
                correction=correction, uncertain_indices=uncertain,
                exact_rows=len(uncertain), exact_fraction=len(uncertain)/len(rows))
