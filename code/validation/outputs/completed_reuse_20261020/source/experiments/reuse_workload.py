"""Proof workload with changing public centres: Stage 1 once vs. Stage 1 per centre set.

One owner catalog (N held-out rows, quantized once). T = 1..10 public centre sets
of the same k: the deployed k-means centres, two refits, two subsample model
updates, Lloyd steps 1/3/10 from k-means++ and two perturbations (training split
only). Every set audits the whole catalog in subset proofs of size b.

  reuse methods (Stage 1 once; one reuse-query setup; one proof per subset per set)
    learned_energy  basis = deployed centre span + PCA residual directions, width 15, + energy
    pca_energy      basis = 15 PCA directions, + energy
    span_energy     basis = deployed centre span only (k-1), + energy
  exact methods (new model per centre set: prepare + Stage 1 + query again)
    cached, affine
  full            no Stage 1; new setup per set; batch b_full measured on one subset,
                  catalog total RECONSTRUCTED as per-proof time x N/b_full (flagged)

Assignments come from the proved witnesses; agreement is against floating Full on
the original points. Reuse proofs also prove a margin sum (soft output).

python -m experiments.reuse_workload --bin zk/bin/commitjl-bound-reuse-linux --out results/reuse_workload_<date> \
    --dataset mnist --n 64
"""
from __future__ import annotations

import argparse
import json
import secrets
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from commitjl.enriched_jl import enrich_centroid_basis
from commitjl.protected_jl import centroid_basis
from data.loaders import load_dataset
from experiments.bound_audit import bounded, save
from experiments.new_center_reuse import center_sets, sqdist

T = 4096
WIDTH = 15                      # head values; + energy = 16 stored statistics
SET_NAMES = ['same', 'refit1', 'refit2', 'subsample1', 'subsample2', 'lloyd1', 'lloyd3', 'lloyd10',
             'perturb0.1', 'perturb0.3']
LOADERS = {'mnist': ('mnist', dict(n_max=5000, seed=0)),
           'cifar10_resnet50': ('cifar10_resnet50', dict(n_max=5000, seed=0))}
LIMIT = 1_600_000


def model_dict(method, d, k, n, scale, centers, basis=()):
    return dict(version='bound-audit-v1', method=method, n=n, d=d, k=k, input_bits=12, basis_scale=T,
                input_scale=format(scale, '.17g'), centers=centers, basis=[list(map(int, r)) for r in basis],
                tail_rows=[], tail_signs=[], tail_width=0, seed='')


def chunks(n, b):
    return [list(range(s, min(s + b, n))) for s in range(0, n, b)]


class Runner:
    def __init__(self, binary, root, reps, warmup):
        self.binary, self.root, self.reps, self.warmup = str(binary), root, reps, warmup
        self.rows = []

    def phase(self, case, phase, inp, q=1, extra=()):
        directory = self.root/'cases'/case
        directory.mkdir(parents=True, exist_ok=True)
        label = f'{phase}_{q}' if phase in ('query', 'reuse') else phase
        cmd = [self.binary, '-phase', phase, '-instance', str(inp), '-out', str(directory), '-q', str(q),
               '-reps', str(self.reps), '-warmup', str(self.warmup), '-max-constraints', str(LIMIT), *extra]
        res = bounded(cmd, directory/f'{label}.process.log', seconds=6*3600)
        out = directory/f'{label}.json'
        result = json.loads(out.read_text()) if out.exists() else dict(status='process_failed')
        if res['stop'] or res['returncode'] not in (0, 3) or not result.get('complete'):
            raise RuntimeError(f'{case}/{label}: {result.get("status")} {result.get("error")} {res}')
        print(json.dumps(dict(case=case, phase=label, constraints=result.get('constraints'),
                              wall=round(res['wall_seconds'], 1))), flush=True)
        return result, res


def proof_seconds(sample):
    return float(np.median(sample['prove_seconds'])) + sample.get('witness_seconds', 0.)


def stage_cost(result):
    return dict(compile=result['compile_seconds'], setup=result['setup_seconds'],
                prove=sum(proof_seconds(s) for s in result['samples']), constraints=result['constraints'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--dataset', choices=sorted(LOADERS), default='mnist')
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--b', type=int, default=64)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--sets', type=int, default=len(SET_NAMES))
    ap.add_argument('--methods', nargs='*', default=['learned_energy', 'pca_energy', 'span_energy',
                                                       'cached', 'affine', 'full'])
    a = ap.parse_args()
    root = a.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    run = Runner(a.bin.resolve(), root, a.reps, 1)

    loader, kw = LOADERS[a.dataset]
    x = np.asarray(load_dataset(loader, **kw)['X'], float)
    x = x[np.linalg.norm(x, axis=1) > 0]
    order = np.random.default_rng(10_000).permutation(len(x))
    tr, au = order[:int(.6*len(x))], order[int(.6*len(x)):]
    xtr = x[tr]
    c0 = KMeans(a.k, n_init=5, random_state=0).fit(xtr).cluster_centers_
    named = {s: c for s, _, c in center_sets(xtr, c0, a.k, 0)[0]}
    sets = [(s, named[s]) for s in SET_NAMES[:a.sets]]
    scale = 4096/max(float(np.abs(x).max()), float(np.abs(c0).max()), 1e-12)
    xi = np.rint(x*scale).astype(np.int64)
    q_sets = [(s, np.clip(np.rint(c*scale), -4096, 4096).astype(np.int64)) for s, c in sets]
    ci = q_sets[0][1]
    catalog = xi[au[:a.n]]
    points = x[au[:a.n]]
    queries = chunks(a.n, min(a.b, a.n))
    d = x.shape[1]
    salt_x, salt_z = str(secrets.randbelow(2**240)+1), str(secrets.randbelow(2**240)+1)
    save(root/'protocol.json', dict(dataset=a.dataset, n=a.n, k=a.k, b=a.b, d=d, reps=a.reps, sets=[s for s, _ in sets],
                                    methods=a.methods, scale=scale, width=WIDTH, limit=LIMIT,
                                    full='one measured subset per set; catalog total reconstructed'))
    refs = []
    for s, c in sets:
        dist = sqdist(points, c)
        lab = dist.argmin(1)
        refs.append((lab, float(dist[np.arange(len(lab)), lab].sum())))
    rows = []

    def record(method, s_idx, kind, cost, labels=None, psi=None, units=None, **info):
        row = dict(dataset=a.dataset, n=a.n, k=a.k, method=method, set=s_idx, set_name=sets[s_idx][0],
                   kind=kind, **cost, **info)
        if labels is not None:
            ref, phi = refs[s_idx]
            row['agreement'] = float(np.mean(np.asarray(labels) == ref))
            row['cost_rel_error'] = float(psi)/units/phi - 1 if psi is not None else np.nan
        rows.append(row)
        pd.DataFrame(rows).to_csv(root/'workload.csv', index=False)

    def assemble(samples):
        by = {}
        for smp in samples:
            by.setdefault(smp.get('centre_set', 0), []).append(smp)
        return by

    # Reuse methods: Stage 1 once, one reuse setup for all sets.
    bases = {}
    if 'learned_energy' in a.methods:
        u, _ = enrich_centroid_basis(xi[tr].astype(float), ci.astype(float), extra=WIDTH - centroid_basis(ci.astype(float)).shape[1],
                                     random_state=0)
        bases['learned_energy'] = u.T
    if 'pca_energy' in a.methods:
        bases['pca_energy'] = PCA(WIDTH, svd_solver='randomized', random_state=0).fit(xi[tr].astype(float)).components_
    if 'span_energy' in a.methods:
        bases['span_energy'] = centroid_basis(ci.astype(float)).T
    for method, basis in bases.items():
        case = f'{a.dataset}_n{a.n}_{method}'
        model = model_dict('learned_energy', d, a.k, a.n, scale, ci.tolist(), np.rint(T*basis))
        inp = root/'cases'/case/'input.json'
        save(inp, dict(model=model, x=catalog.tolist(), salt_x=salt_x, salt_z=salt_z, queries=queries,
                       reuse_centers=[c.tolist() for _, c in q_sets]))
        prep, _ = run.phase(case, 'prepare', inp)
        st, res1 = run.phase(case, 'stage1', inp)
        record(method, 0, 'stage1', dict(stage_cost(st), native=prep['native_prepare_seconds'],
                                          peak_rss=res1['peak_rss_bytes']), basis_rank=len(basis))
        rq, resq = run.phase(case, 'reuse', inp, q=len(queries[0]))
        by = assemble(rq['samples'])
        for s_idx in range(len(sets)):
            smp = by.get(s_idx, [])
            labels = [lab for sm in sorted(smp, key=lambda z: z['id']) for lab in sm['assign']]
            psi = sum(int(sm['psi']) for sm in smp)
            margin = sum(int(sm['margin_sum']) for sm in smp)
            cost = dict(prove=sum(proof_seconds(sm) for sm in smp), constraints=rq['constraints'],
                        compile=rq['compile_seconds'] if s_idx == 0 else 0., setup=rq['setup_seconds'] if s_idx == 0 else 0.,
                        peak_rss=resq['peak_rss_bytes'])
            record(method, s_idx, 'query', cost, labels, psi, float(T)**4*scale**2,
                   margin_sum=margin/float(T)**4/scale**2, basis_rank=len(basis))

    # Exact methods: everything again per centre set.
    for method in [m for m in ('cached', 'affine') if m in a.methods]:
        for s_idx, (s, c) in enumerate(q_sets):
            case = f'{a.dataset}_n{a.n}_{method}_set{s_idx:02d}'
            model = model_dict(method, d, a.k, a.n, scale, c.tolist())
            inp = root/'cases'/case/'input.json'
            save(inp, dict(model=model, x=catalog.tolist(), salt_x=salt_x, salt_z=salt_z, queries=queries))
            prep, _ = run.phase(case, 'prepare', inp)
            st, res1 = run.phase(case, 'stage1', inp)
            record(method, s_idx, 'stage1', dict(stage_cost(st), native=prep['native_prepare_seconds'],
                                                  peak_rss=res1['peak_rss_bytes']))
            rq, resq = run.phase(case, 'query', inp, q=len(queries[0]))
            smp = sorted(rq['samples'], key=lambda z: z['id'])
            labels = [lab for sm in smp for lab in sm['assign']]
            record(method, s_idx, 'query', dict(stage_cost(rq), peak_rss=resq['peak_rss_bytes']), labels,
                   sum(int(sm['psi']) for sm in smp), scale**2)

    # Full: original rows; one measured subset per set, reconstructed catalog total.
    if 'full' in a.methods:
        bf = 1
        while bf*2 <= a.n and bf*2*215*d <= 1_450_000:
            bf *= 2
        for s_idx, (s, c) in enumerate(q_sets):
            case = f'{a.dataset}_n{a.n}_full_set{s_idx:02d}'
            model = model_dict('full', d, a.k, a.n, scale, c.tolist())
            inp = root/'cases'/case/'input.json'
            save(inp, dict(model=model, x=catalog.tolist(), salt_x=salt_x, salt_z=salt_z, queries=[queries[0][:bf]]))
            run.phase(case, 'prepare', inp)
            rq, resq = run.phase(case, 'query', inp, q=bf)
            per = proof_seconds(rq['samples'][0])
            record('full', s_idx, 'query', dict(compile=rq['compile_seconds'], setup=rq['setup_seconds'],
                                                prove=per*a.n/bf, constraints=rq['constraints'],
                                                peak_rss=resq['peak_rss_bytes']),
                   reconstructed=True, full_batch=bf, measured_proof_seconds=per)
    save(root/'done.json', dict(complete=True, finished_unix=time.time()))


if __name__ == '__main__':
    main()
