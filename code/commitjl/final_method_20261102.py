"""Frozen conservative rank selection for reusable prototype-update proofs."""
from dataclasses import asdict
import time

import numpy as np
from sklearn.decomposition import PCA

from commitjl.calibrated_routing_20261102 import BatchCosts, choose_routes
from commitjl.enriched_jl import enrich_centroid_basis
from commitjl.protected_jl import centroid_basis
from experiments.pca_advantage_20261102 import SCALE

RANKS = (16, 79)
RANK_MARGIN = .10


def fit_bases(past, old, seed):
    """Past training rows only; no history-mixture search in the final method."""
    bases, seconds = {}, {}
    started = time.perf_counter()
    bases['pca'] = PCA(79, svd_solver='randomized', iterated_power=4,
                       random_state=seed).fit(past.astype(float)).components_
    seconds['pca'] = time.perf_counter()-started
    started = time.perf_counter()
    protected = centroid_basis(old)
    bases['centroid'] = enrich_centroid_basis(past.astype(float), old,
        extra=79-protected.shape[1], random_state=seed, iterated_power=4,
        preserve_pca_order=True)[0].T
    seconds['centroid'] = time.perf_counter()-started
    for name, basis in bases.items():
        if basis.shape != (79, past.shape[1]) or not np.allclose(basis@basis.T, np.eye(79), atol=1e-9):
            raise ValueError('Invalid fitted basis')
        bases[name] = np.rint(basis*SCALE).astype(np.int64)
    return bases, seconds


def validation_choice(records, family, profiles):
    """Freeze before final evaluation; identical rule for centroid and PCA.

    Lower rank needs all eight validation updates conclusive, a measured
    compatible profile, and ten percent lower predicted eight-request service
    cost including authentication, lazy refinement and Full initialization.
    """
    candidates = {}
    for rank in RANKS:
        rows = [r for r in records if r['family'] == family and r['rank'] == rank]
        if len(rows) != 8 or {r['request'] for r in rows} != set(range(8)):
            raise ValueError('Need exactly eight distinct validation updates per rank')
        profile = profiles.get(str(rank))
        if profile is None:
            candidates[str(rank)] = dict(eligible=False, reason='No compatible measured profile')
            continue
        cost = BatchCosts(**profile['costs'])
        work = [dict(request_id=r['request'], conclusive=r['conclusive'],
                     predicted_corrections=r['corrections']) for r in sorted(rows, key=lambda r:r['request'])]
        plan = choose_routes(work, cost)
        setup = profile['prepare_seconds']
        if plan['certificate']:
            setup += profile['keys_rows_seconds']
        if any(work[i]['predicted_corrections'] for i in plan['certificate']):
            setup += profile['refinement_setup_seconds']
        if plan['full']:
            setup += profile['full_initialization_seconds']
        candidates[str(rank)] = dict(eligible=rank == 79 or all(r['conclusive'] for r in rows),
             conclusive=sum(r['conclusive'] for r in rows), corrections=sum(r['corrections'] for r in rows),
             predicted_service_seconds=setup+plan['predicted_routed_seconds'],
             predicted_warm_seconds=plan['predicted_routed_seconds'], plan=plan)
    if not candidates['79'].get('eligible'):
        raise ValueError('A measured default-rank profile is required')
    small, default = candidates['16'], candidates['79']
    selected = 16 if (small.get('eligible') and small['predicted_service_seconds'] <
                      default['predicted_service_seconds']*(1-RANK_MARGIN)) else 79
    return dict(family=family, rank=selected, candidates=candidates,
                rule='Eight of eight validation updates conclusive and at least10% predicted service saving; otherwise79.')
