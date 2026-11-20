"""Exact, dependency-free checks of the added analysis, not a new audit method."""
from __future__ import annotations

import argparse
import json
from math import isqrt, sqrt
from pathlib import Path
import random


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def sub(a, b):
    return [x-y for x, y in zip(a, b)]


def ceiling_norm(a):
    v = dot(a, a)
    root = isqrt(v)
    return root + (root*root < v)


def distance(a, b):
    return dot(sub(a, b), sub(a, b))


def cost(rows, centers):
    return sum(min(distance(row, c) for c in centers) for row in rows)


def analyze(rows, old, new, basis, scale, budget=0):
    """Independently evaluate the existing affine formula with Python integers."""
    t2 = scale**2
    anchor = old[0]

    def residual(v, anchored=True):
        w = sub(v, anchor) if anchored else list(v)
        bw = [dot(b, w) for b in basis]
        return [t2*w[j]-sum(bw[a]*basis[a][j] for a in range(len(basis)))
                for j in range(len(w))]

    def stats(v):
        w = sub(v, anchor)
        return [scale*dot(b, w) for b in basis], dot(residual(v), residual(v))

    gram = [[dot(b, c)-(t2 if i == j else 0) for j, c in enumerate(basis)]
            for i, b in enumerate(basis)]

    def quadratic(v):
        return sum(v[i]*dot(g, v) for i, g in enumerate(gram))

    groups = [min(range(len(old)), key=lambda j: distance(x, old[j])) for x in rows]
    assigned = sum(distance(x, new[g])-distance(x, old[g]) for x, g in zip(rows, groups))
    deltas = [sub(b, a) for a, b in zip(old, new)]
    lower, upper, exact = [], [], []
    direct_bounds, drift_bounds = [], []
    for x, g in zip(rows, groups):
        h, _ = stats(x)
        hg, eg = stats(new[g])
        rho = ceiling_norm(residual(x))
        pairs = []
        max_tau = max_drift = 0
        for j, c in enumerate(new):
            hj, ej = stats(c)
            dh = sub(hj, hg)
            q = 2*sum(h[a]*(dot(gram[a], dh)-t2*dh[a]) for a in range(len(basis)))
            q += t2*(dot(hj, hj)-dot(hg, hg)+ej-eg)-quadratic(hj)+quadratic(hg)
            rdiff = sub(residual(c), residual(new[g]))
            tau = ceiling_norm(rdiff)
            alpha = ceiling_norm(residual(sub(old[j], old[g]), False))
            gamma = ceiling_norm(residual(sub(deltas[j], deltas[g]), False))
            assert tau <= alpha+gamma
            f = scale**6*(distance(x, c)-distance(x, new[g]))
            assert f == q-2*t2*dot(residual(x), rdiff), 'rounded-basis identity'
            error = 2*t2*rho*tau
            assert q-error <= f <= q+error
            pairs.append((q-error, q+error))
            max_tau, max_drift = max(max_tau, tau), max(max_drift, alpha+gamma)
        lo, hi = min(a for a, _ in pairs), min(b for _, b in pairs)
        value = scale**6*(min(distance(x, c) for c in new)-distance(x, new[g]))
        assert lo <= value <= hi <= 0
        assert hi-lo <= 4*t2*rho*max_tau <= 4*t2*rho*max_drift
        lower.append(lo); upper.append(hi); exact.append(value)
        direct_bounds.append(4*t2*rho*max_tau)
        drift_bounds.append(4*t2*rho*max_drift)
    order = sorted(range(len(rows)), key=lambda i: (-(upper[i]-lower[i]), i))
    selected = set(order[:budget])
    lo = scale**6*assigned+sum(exact[i] if i in selected else lower[i] for i in range(len(rows)))
    hi = scale**6*assigned+sum(exact[i] if i in selected else upper[i] for i in range(len(rows)))
    width = sum(upper[i]-lower[i] for i in range(len(rows)) if i not in selected)
    bound = sum(drift_bounds[i] for i in range(len(rows)) if i not in selected)
    truth = cost(rows, new)-cost(rows, old)
    assert lo <= scale**6*truth <= hi
    assert hi-lo == width <= bound
    margin_sufficient = abs(scale**6*truth) > bound
    if margin_sufficient:
        assert lo > 0 or hi < 0
    assert (abs(lo+hi) > hi-lo) == (lo > 0 or hi < 0)
    return dict(lower=lo, upper=hi, width=width, drift_width_bound=bound,
                denominator=scale**6, truth=truth, assigned=assigned,
                corrected=len(selected), margin_sufficient=margin_sufficient,
                row_lower=lower, row_upper=upper, groups=groups)


def run(core=False):
    rng = random.Random(20261030)
    checks = margin_cases = comparisons = 0
    for case in range(160):
        d, k, n = rng.randint(2, 7), rng.randint(2, 4), rng.randint(3, 12)
        s, t = rng.randint(1, d), [1, 2, 8, 16, 16384][case % 5]
        orthogonal = []
        for _ in range(s):
            v = [rng.gauss(0, 1) for _ in range(d)]
            for b in orthogonal:
                coeff = dot(v, b)
                v = [a-coeff*c for a, c in zip(v, b)]
            norm = sqrt(dot(v, v))
            orthogonal.append([a/norm for a in v])
        basis = [[round(t*v) for v in b] for b in orthogonal]
        old = [[rng.randint(-8, 8) for _ in range(d)] for _ in range(k)]
        new = [[v+rng.randint(-4, 4) for v in c] for c in old]
        rows = [[rng.randint(-16, 16) for _ in range(d)] for _ in range(n)]
        for budget in sorted({0, 1, n//2, n}):
            result = analyze(rows, old, new, basis, t, budget)
            checks += 1; margin_cases += result['margin_sufficient']
        if core:
            from commitjl.group_change_bounds import exact_old_correction_bounds
            from commitjl.group_moment_reference import prepare_groups
            model = dict(centers=old, basis=basis, basis_scale=t)
            prepared = prepare_groups(rows, result['groups'], k)
            native = exact_old_correction_bounds(model, rows, prepared, old, new)
            assert list(map(int, native['row_lower'])) == result['row_lower']
            assert list(map(int, native['row_upper'])) == result['row_upper']
            assert int(native['assigned_scaled']) == result['assigned']*t**6
            comparisons += 1
    rows, old, basis = [[0, 2]]*128, [[0, 0], [10, 0]], [[1, 0]]
    favorable = analyze(rows, old, [[0, 1], [10, 1]], basis, 1)
    failure0 = analyze(rows, old, [[0, -2], [0, 1]], basis, 1)
    failure16 = analyze(rows, old, [[0, -2], [0, 1]], basis, 1, 16)
    failure102 = analyze(rows, old, [[0, -2], [0, 1]], basis, 1, 102)
    failure103 = analyze(rows, old, [[0, -2], [0, 1]], basis, 1, 103)
    assert (favorable['lower'], favorable['upper'], favorable['truth']) == (-384, -384, -384)
    assert (failure0['lower'], failure0['upper']) == (-384, 1536)
    assert (failure16['lower'], failure16['upper'], failure16['truth']) == (-384, 1296, -384)
    assert failure102['upper'] >= 0 and failure103['upper'] < 0
    compact = lambda r: {key: val for key, val in r.items() if not key.startswith('row_') and key != 'groups'}
    return dict(complete=True, random_seed=20261030, rounded_basis_cases=160,
                interval_budget_checks=checks, sufficient_margin_cases=margin_cases,
                existing_core_comparisons=comparisons, favorable=compact(favorable),
                failure_initial=compact(failure0), failure_cap16=compact(failure16),
                first_strict_improvement_correction_count=103,
                new_performance_measurements=False, method_changed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', action='store_true', help='Also compare with the unchanged NumPy implementation')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args.core)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))
