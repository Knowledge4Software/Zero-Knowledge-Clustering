"""Finite paper controls: matched ablations, precision, baselines and larger catalogs.

Existing evidence is read or copied, never modified. Public-proof exports exclude
private arrays, salts, assignments and proving keys. No result-dependent selection.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path
import secrets
import shutil
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.bound_audit import DATASETS, save
from experiments.paper_overnight import dataset, seed_for, run_bounded
from commitjl.protected_jl import centroid_basis
from commitjl.reusable_native import stable_squared_distances
from commitjl.sparse_gen import sparse_matrix_gen_csr

T = 4096
VARIANTS = ('head_only', 'protected_linear', 'sparse_jl', 'rademacher', 'pca_equal_stats', 'cached')


def split_fit(x, k, start):
    order = np.random.default_rng(10000+start).permutation(len(x))
    tr, au = order[:int(.6*len(x))], order[int(.6*len(x)):]
    c = KMeans(k, n_init=5, random_state=start).fit(x[tr]).cluster_centers_
    return tr, au, c


def quality(x, c, labels, costs, ref, phi):
    assigned = np.sum((x-c[labels])**2, axis=1)
    return dict(ari=float(adjusted_rand_score(ref, labels)), agreement=float(np.mean(ref==labels)),
                r_phi=float(assigned.sum()/phi), cost_relative_error=float(costs.sum()/phi-1),
                cost_abs_relative_error=float(abs(costs.sum()/phi-1)))


def native_worker(a):
    x, meta = dataset(a.dataset, smoke=a.smoke)
    tr, au, c = split_fit(x, a.k, a.start)
    xa = x[au]; full_d = stable_squared_distances(xa, c); ref = full_d.argmin(1)
    phi = float(full_d[np.arange(len(xa)), ref].sum())
    scale = 4096/max(np.abs(x).max(), np.abs(c).max(), 1e-12)
    xi, ci = np.rint(xa*scale), np.rint(c*scale)
    w, delta = xi-ci[0], ci-ci[0]
    qreference = stable_squared_distances(xi, ci).argmin(1)
    ts = time.perf_counter(); u = np.rint(T*centroid_basis(ci).T)/T; basis_fit = time.perf_counter()-ts
    ts = time.perf_counter(); h, hc = w@u.T, delta@u.T; head_apply = time.perf_counter()-ts
    ts = time.perf_counter(); residual, rc = w-h@u, delta-hc@u; residual_apply = time.perf_counter()-ts
    ts = time.perf_counter(); e = np.einsum('ij,ij->i', residual, residual); energy_apply = time.perf_counter()-ts
    raw = dict(train_indices=tr, audit_indices=au, centers=c, quantized_centers=ci, basis=u,
               floating_labels=ref, floating_costs=full_d[np.arange(len(xa)), ref])
    rows, exclusions = [], []
    def record(label, z, zc, energy=None, *, budget=0, map_id=0, q=0, policy='fixed_budget',
               fit=0., apply=0., input_bits=12, basis_bits=12):
        t = time.perf_counter(); d = stable_squared_distances(z, zc); labels = d.argmin(1)
        costs = d[np.arange(len(z)), labels]+(energy if energy is not None else 0.)
        downstream = time.perf_counter()-t
        key = f'{label}_m{budget}_map{map_id}_q{q}_{policy}_i{input_bits}_b{basis_bits}'
        raw[key+'_labels'], raw[key+'_costs'] = labels, costs/scale**2
        rows.append(dict(dataset=a.dataset, k=a.k, start=a.start, method=label, budget=budget,
                         map=map_id, q=q, policy=policy, input_bits=input_bits, basis_bits=basis_bits,
                         n=len(x), n_audit=len(xa), d=x.shape[1], stored_columns=z.shape[1]+int(energy is not None),
                         model_fit_seconds=fit, compression_seconds=apply, downstream_seconds=downstream,
                         ari_quantized_full=float(adjusted_rand_score(qreference,labels)),
                         agreement_quantized_full=float(np.mean(qreference==labels)),
                         **quality(xa, c, labels, costs/scale**2, ref, phi)))
    record('full', xi, ci, budget=x.shape[1])
    ts=time.perf_counter();scores=2*(w@delta.T)-(delta*delta).sum(1)
    cached_labels=scores.argmax(1)
    cached_costs=np.einsum('ij,ij->i',w,w)-scores[np.arange(len(w)),cached_labels]
    cached_prepare=time.perf_counter()-ts
    ts=time.perf_counter();float(cached_costs.sum());cached_query=time.perf_counter()-ts
    raw['cached_labels'],raw['cached_costs']=cached_labels,cached_costs/scale**2
    rows.append(dict(dataset=a.dataset,k=a.k,start=a.start,method='cached',budget=2,map=0,q=0,
        policy='fixed_task_answers',input_bits=12,basis_bits=0,n=len(x),n_audit=len(xa),d=x.shape[1],stored_columns=2,
        model_fit_seconds=0.,compression_seconds=cached_prepare,downstream_seconds=cached_query,
        ari_quantized_full=float(adjusted_rand_score(qreference,cached_labels)),agreement_quantized_full=float(np.mean(qreference==cached_labels)),
        **quality(xa,c,cached_labels,cached_costs/scale**2,ref,phi)))
    record('head_only', h, hc, budget=len(u), fit=basis_fit, apply=head_apply)
    record('head_energy', h, hc, e, budget=len(u)+1, fit=basis_fit, apply=head_apply+residual_apply+energy_apply)
    budgets = [16] if a.smoke else [16, 32, 64]
    # Match PCA to the actual h+1 protected statistics as well as conventional widths.
    equal_stats = len(u)+1
    if equal_stats not in budgets and equal_stats <= min(len(tr),x.shape[1]):
        t=time.perf_counter(); p=PCA(equal_stats,svd_solver='randomized',random_state=a.start).fit(x[tr])
        b=np.rint(T*p.components_)/T;fit=time.perf_counter()-t
        t=time.perf_counter();z,zc=w@b.T,delta@b.T;elapsed=time.perf_counter()-t
        record('pca',z,zc,budget=equal_stats,fit=fit,apply=elapsed)
    maps = 1 if a.smoke else 5
    for m in budgets:
        if m <= min(len(tr), x.shape[1]):
            t = time.perf_counter(); p = PCA(m, svd_solver='randomized', random_state=a.start).fit(x[tr])
            b = np.rint(T*p.components_)/T; fit = time.perf_counter()-t
            t = time.perf_counter(); z, zc = w@b.T, delta@b.T; elapsed = time.perf_counter()-t
            record('pca', z, zc, budget=m, fit=fit, apply=elapsed)
        else: exclusions.append(dict(method='pca',budget=m,reason='training/dimension rank restriction'))
        for s in range(maps):
            rng = np.random.default_rng(int.from_bytes(seed_for('paper-sign-v1', a.dataset, a.k, a.start, m, s), 'big'))
            t = time.perf_counter(); b = np.rint(T*rng.choice([-1., 1.], size=(m, x.shape[1]))/np.sqrt(m))/T
            fit = time.perf_counter()-t
            t = time.perf_counter(); z, zc = w@b.T, delta@b.T; elapsed = time.perf_counter()-t
            record('rademacher', z, zc, budget=m, map_id=s, fit=fit, apply=elapsed)
            for q in ([1] if a.smoke else [1, 2, 4]):
                seed = seed_for('paper-jl-v1', a.dataset, a.k, a.start, m, s, q)
                t = time.perf_counter(); _, mat = sparse_matrix_gen_csr(seed, m, x.shape[1], q)
                fit = time.perf_counter()-t
                t = time.perf_counter(); z, zc = (mat@w.T).T/np.sqrt(q), (mat@delta.T).T/np.sqrt(q)
                elapsed = time.perf_counter()-t
                record('sparse_jl', z, zc, budget=m, map_id=s, q=q, fit=fit, apply=elapsed)
            # Same head and SAME random tail: removal of E changes no other component.
            width = m-len(u)-1
            if width < 1:
                exclusions.append(dict(method='head_jl',budget=m,reason='protected rank plus energy exhausts budget'))
                continue
            seed = seed_for('paper-residual-v1', a.dataset, a.k, a.start, m, s)
            t = time.perf_counter(); _, mat = sparse_matrix_gen_csr(seed, width, x.shape[1], 1)
            fit = time.perf_counter()-t
            t = time.perf_counter(); tail, tc = (mat@residual.T).T, (mat@rc.T).T
            z, zc = np.column_stack((h, tail)), np.column_stack((hc, tc)); elapsed = time.perf_counter()-t
            record('head_jl', z, zc, budget=m, map_id=s, q=1, policy='fixed_components',
                   fit=basis_fit+fit, apply=head_apply+residual_apply+elapsed)
            # Energy query intentionally uses head only; tail is stored for separate geometry queries.
            record('head_energy_jl', h, hc, e, budget=m, map_id=s, q=1, policy='fixed_components',
                   fit=basis_fit+fit, apply=head_apply+residual_apply+energy_apply+elapsed)
            rows[-1]['stored_columns'] = len(u)+1+width
            t = time.perf_counter(); _, mat = sparse_matrix_gen_csr(seed, width+1, x.shape[1], 1); fit = time.perf_counter()-t
            t = time.perf_counter(); z = np.column_stack((h, (mat@residual.T).T)); zc = np.column_stack((hc, (mat@rc.T).T))
            elapsed = time.perf_counter()-t
            record('head_jl', z, zc, budget=m, map_id=s, q=1, policy='fixed_budget',
                   fit=basis_fit+fit, apply=head_apply+residual_apply+elapsed)
    # Precision uses the same floating centers and split, not a newly selected model.
    for ib in ([12] if a.smoke else [8, 12, 16]):
        quant = 2**ib/max(np.abs(x).max(), np.abs(c).max(), 1e-12)
        xx, cc = np.rint(xa*quant), np.rint(c*quant); ww, dd = xx-cc[0], cc-cc[0]
        for bb in ([12] if a.smoke else [8, 12, 16]):
            b = np.rint(2**bb*centroid_basis(cc).T)/2**bb
            hh, hcc = ww@b.T, dd@b.T; rr = ww-hh@b; ee = np.einsum('ij,ij->i', rr, rr)
            old_scale, old_reference = scale, qreference; scale = quant
            qreference = stable_squared_distances(xx, cc).argmin(1)
            record('precision_head_energy', hh, hcc, ee, budget=len(b)+1, input_bits=ib, basis_bits=bb, policy='precision')
            scale, qreference = old_scale, old_reference
    np.savez_compressed(a.out/'raw.npz', **raw)
    save(a.out/'result.json', dict(status='complete', complete=True, task='native', dataset=meta,
        rows=rows, exclusions=exclusions, note='Original floating Full is reference; fixed-component and fixed-budget comparisons are distinct.'))


def subsets(n, large=False):
    rng = np.random.default_rng(429)
    out = []
    for q in ([1, 8, 32, 64] if large else [1, 8]):
        if q > n: continue
        target = 1 if q==n else 2 if q==1 else 4
        seen = set()
        for _ in range(1000):
            seen.add(tuple(sorted(rng.choice(n, q, replace=False).tolist())))
            if len(seen) == target: break
        if len(seen)!=target: raise RuntimeError('finite subset sampler did not reach target')
        out += [list(v) for v in sorted(seen)]
    return out


def public_export(bundle, out):
    """Whitelist public verification artifacts; never copy a whole prover folder."""
    out.mkdir()
    paths = ['trusted.json', 'prepare.json', 'stage1.json', 'geometry.json']
    paths += [p.name for p in bundle.glob('query_*.json') if p.stem.removeprefix('query_').isdigit()]
    files = []
    for name in paths:
        p = bundle/name
        if not p.exists(): continue
        value = json.loads(p.read_text())
        for sample in value.get('samples') or []: sample.pop('assign', None)
        save(out/name, value); files.append(name)
    for pattern in ['keys/*.vk', 'stage1/*.proof', 'queries/*.proof', 'geometry/*.proof']:
        for p in bundle.glob(pattern):
            target = out/p.relative_to(bundle); target.parent.mkdir(exist_ok=True)
            shutil.copyfile(p, target); files.append(str(p.relative_to(bundle)))
    save(out/'public_manifest.json', dict(files=files,
        disclosure='Public centers/model, roots, dimensions/scales, query indices and exact claimed costs/distances. No DP claim.',
        trust='Owner root and verification keys must be obtained through a trusted channel.',
        sha256={f: hashlib.sha256((out/f).read_bytes()).hexdigest() for f in files}))


def proof_worker(a):
    source = a.source/'cases'/f'{a.dataset}_k{a.k}_protected_tail'
    if a.smoke: source = a.source/'cases'/'synthetic_smoke_k5_protected_tail'
    if not json.loads((source/'verify.json').read_text()).get('complete'): raise RuntimeError('source not independently verified')
    inp = json.loads((source/'input.json').read_text()); model = copy.deepcopy(inp['model'])
    ctx = json.loads((source/'native_context.json').read_text())
    n = model['n']; original_root = json.loads((source/'prepare.json').read_text())['statement']['cx']
    model_fit = 0.
    if a.large:
        x, _ = dataset(a.dataset, smoke=a.smoke); tr, au, c = split_fit(x, a.k, 0)
        scale = float(model['input_scale']); n = 4 if a.smoke else 64
        inp.update(x=np.rint(x[au[:n]]*scale).astype(np.int64).tolist(), salt_x=str(secrets.randbelow(2**240)+1), salt_z=str(secrets.randbelow(2**240)+1))
        # The catalog is NEW, but salts are shared across all methods via catalog.json.
        shared = a.out.parents[1]/f'catalog_{a.dataset}_k{a.k}.json'
        if shared.exists(): inp.update(json.loads(shared.read_text()))
        else: save(shared, {k: inp[k] for k in ['x', 'salt_x', 'salt_z']})
        original_root = None
    model['n'] = n
    inp['queries'] = subsets(n, a.large)
    clean = dict(tail_width=0, tail_rows=[], tail_signs=[])
    if a.variant=='head_only': model.update(method='pca', **clean)
    elif a.variant=='protected_linear': model['method']='protected_linear'
    elif a.variant in ['sparse_jl', 'rademacher', 'pca_equal_stats']:
        m = a.k if a.variant=='pca_equal_stats' else a.budget
        nonce = secrets.token_hex(32)
        seed = seed_for('supplement-public-map-v1', a.dataset, a.k, a.variant, m, original_root or 'new-catalog', nonce)
        save(a.out/'map_recipe.json', dict(owner_root=original_root, variant=a.variant, width=m, nonce=nonce, seed=seed.hex(),
            note='Host-generated public map; original root was already fixed. No in-circuit generator or beacon claim.'))
        start = time.perf_counter()
        if a.variant=='sparse_jl':
            _, matrix = sparse_matrix_gen_csr(seed, m, model['d'], 1); b = T*matrix.toarray()
        elif a.variant=='rademacher':
            rng = np.random.default_rng(int.from_bytes(seed,'big')); b = np.rint(T*rng.choice([-1, 1], (m, model['d']))/np.sqrt(m))
        else:
            x, _ = dataset(a.dataset, smoke=a.smoke); tr = np.random.default_rng(10000).permutation(len(x))[:int(.6*len(x))]
            b = np.rint(T*PCA(m, svd_solver='randomized', random_state=0).fit(x[tr]).components_)
        model_fit = time.perf_counter()-start
        model.update(method='pca', basis=b.astype(np.int64).tolist(), seed=seed.hex(), **clean)
    elif a.variant in ['full', 'affine', 'cached']:
        model.update(method=a.variant, basis=[], **clean)
    elif a.variant=='pca':
        control = a.source/'cases'/f'{a.dataset}_k{a.k}_pca'/'input.json'
        if a.smoke: control = a.source/'cases'/'synthetic_smoke_k5_pca'/'input.json'
        model.update(json.loads(control.read_text())['model']); model['n']=n
    elif a.variant=='protected_energy': model.update(method='protected_energy', **clean)
    elif a.variant!='protected_tail': raise RuntimeError('unknown variant')
    inp['model'] = model
    input_path = a.out/'private_input.json'; save(input_path, inp)
    bundle = a.out/'bundle'; events = []
    def phase(name, q=None, target=bundle):
        cmd = [str(a.bin), '-phase', name, '-instance', str(input_path), '-out', str(target),
               '-reps', '1' if a.smoke else '3', '-warmup', '0' if a.smoke else '1', '-max-constraints', '1300000']
        if q is not None: cmd += ['-q', str(q)]
        if name=='verify': cmd += ['-trust', str(target/'trusted.json')]
        key = name+(f'_{q}' if q is not None else '')+('_public' if target!=bundle else '')
        resource = run_bounded(cmd, a.out/(key+'.log'), seconds=2400)
        result_path = target/(f'query_{q}.json' if name=='query' else name+'.json')
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        events.append(dict(phase=key, resource=resource, result=result)); save(a.out/'events.json', events)
        if resource['returncode'] not in [0,3] or resource['stop']: raise RuntimeError(f'{key}: {resource}')
        return result
    prep = phase('prepare')
    if original_root and prep['statement']['cx']!=original_root: raise RuntimeError('changed original owner root')
    if a.large:
        shared = a.out.parents[1]/f'catalog_root_{a.dataset}_k{a.k}.json'
        cx = prep['statement']['cx']
        if shared.exists(): assert json.loads(shared.read_text())['cx']==cx
        else: save(shared, dict(cx=cx))
    if model['method']!='full':
        if not phase('stage1').get('complete'):
            save(a.out/'result.json', dict(status='excluded', complete=False, task='proof', reason='preprocessing budget', events=events)); return
    queries = []
    for q in sorted(set(map(len, inp['queries']))):
        # Full q>=8 on MNIST exceeds the already measured constraint budget.
        if model['method']=='full' and q>1: continue
        result = phase('query', q)
        if result.get('complete'): queries.append(result)
    if not queries: raise RuntimeError('no completed query shape')
    verified = phase('verify')
    public_export(bundle, a.out/'public_bundle')
    exported = phase('verify', target=a.out/'public_bundle')
    if not exported.get('complete'): raise RuntimeError('verification required private artifacts')
    # Diagnostic assignments stay in the PRIVATE experiment directory only.
    xf = np.asarray(inp['x'], float); cc = np.asarray(model['centers'], float)
    qref = stable_squared_distances(xf, cc); labels = qref.argmin(1)
    scale = float(model['input_scale']); power = 1 if model['method'] in ['full','affine','cached'] else model['basis_scale']**4
    samples = []
    for query in queries:
        for sample in query['samples']:
            idx = np.asarray(sample['indices']); predicted = np.asarray(sample['assign'])
            true_cost = qref[idx, labels[idx]].sum()/scale**2
            samples.append(dict(q=len(idx), agreement_quantized_full=float(np.mean(predicted==labels[idx])),
                ari_quantized_full=float(adjusted_rand_score(labels[idx], predicted)),
                cost_relative_error=int(sample['psi'])/power/scale**2/true_cost-1 if true_cost else None))
    save(a.out/'result.json', dict(status='complete', complete=True, task='proof', dataset=a.dataset, k=a.k,
        variant=a.variant, budget=a.budget, large=a.large, n_catalog=n, events=events, quality=samples,
        model_fit_seconds=model_fit, independently_verified=verified['complete'], public_export_verified=exported['complete'],
        original_root=prep['statement']['cx'],
        trust='Public map approval and its generation are host duties; internal pca method means a supplied linear map, not proof of PCA training.'))


def summarize(root):
    native, proofs, inventory = [], [], []
    for p in sorted((root/'jobs').glob('*/result.json')):
        v = json.loads(p.read_text()); inventory.append(dict(job=p.parent.name, task=v['task'], status=v['status']))
        if not v.get('complete'): continue
        if v['task']=='native': native.extend(v['rows']); continue
        for event in v['events']:
            res = event['result']; samples = res.get('samples') or []
            times = [np.median(s['prove_seconds']) for s in samples if s.get('prove_seconds')]
            proofs.append(dict(job=p.parent.name, dataset=v['dataset'], k=v['k'], variant=v['variant'], budget=v['budget'],
                large=v['large'], n_catalog=v['n_catalog'], phase=event['phase'], status=res.get('status'),
                constraints=res.get('constraints',0), prove_median_seconds=float(np.median(times)) if times else None,
                compile_seconds=res.get('compile_seconds',0), setup_seconds=res.get('setup_seconds',0),
                prove_sum_seconds=float(np.sum(times)), distinct_instances=len(samples),
                observed_wall_seconds=event['resource']['wall_seconds'],
                public_export_verified=v['public_export_verified']))
    for name, rows in [('native',native),('proofs',proofs),('inventory',inventory)]: pd.DataFrame(rows).to_csv(root/(name+'.csv'), index=False)
    text = ['# Predeclared paper supplement', '', f'Completed receipts: {len(inventory)}.',
        'Native ablations, matched storage controls, q=1/2/4 sparsity and signed dense projection, input/basis precision, and larger-center sweeps.',
        'Actual proofs retain original-data binding, deterministic ties and membership under global roots. All exported public bundles are independently verified without private arrays or assignment metadata.',
        'Costs of original preparation, all preprocessing proofs, compile/setup and genuinely distinct queries must be included. No large-corpus proof, general superiority or journal-acceptance claim follows.',
        'Native results use floating Full. Proof quality uses quantized Full. Different studies must not be pooled blindly.',
        'Files: native.csv, proofs.csv, inventory.csv, jobs/*/raw.npz and private/public proof artifacts.', '']
    (root/'report.md').write_text('\n'.join(text), encoding='utf-8')


def campaign(a):
    if (a.out/'status.json').exists(): raise RuntimeError('fresh output required')
    jobs = []
    if a.smoke:
        jobs = [dict(task='native', dataset='gaussian_hard', k=5, start=0)]
        jobs += [dict(task='proof', dataset='gaussian_hard', k=5, variant=v) for v in VARIANTS]
    else:
        # Complete the missing ZK controls first. All four cells share source roots with prior evidence.
        for name,k in [('gaussian_hard',5),('gaussian_hard',10),('mnist',5),('20ng_tfidf_nonzero',5)]:
            jobs += [dict(task='proof', dataset=name, k=k, variant=v) for v in VARIANTS]
        jobs += [dict(task='native',dataset=name,k=k,start=s) for name,_,_ in DATASETS for k in [5,10] for s in range(3)]
        jobs += [dict(task='native',dataset=name,k=k,start=s) for name in ['gaussian_hard','mnist'] for k in [2,20,50] for s in range(3)]
        jobs += [dict(task='proof',dataset='mnist',k=5,variant=v,large=True) for v in ['full','pca','protected_energy','protected_tail','affine','cached']]
    started=time.time(); state=dict(status='running',complete=False,host=platform.node(),started_unix=started,
        jobs_done=0,jobs_planned=len(jobs),failures=[],exclusions=[],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(a.out/'plan.json',jobs);save(a.out/'status.json',state)
    for i,job in enumerate(jobs):
        label=f'{i:03d}_{job["task"]}_{job["dataset"]}_k{job["k"]}_s{job.get("start",0)}_{job.get("variant", "all")}'
        folder=a.out/'jobs'/label;folder.mkdir(parents=True)
        state.update(current_job=label,elapsed_seconds=time.time()-started);save(a.out/'status.json',state)
        remaining=6*3600-(time.time()-started)
        if remaining<=0: state['status']='deadline';break
        cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',job['task'],'--out',str(folder),'--bin',str(a.bin),'--source',str(a.source)]
        for key in ['dataset','k','start','variant']:
            if key in job:cmd += ['--'+key,str(job[key])]
        if job.get('large'):cmd+=['--large']
        if a.smoke:cmd+=['--smoke']
        resource=run_bounded(cmd,folder/'worker.log',seconds=min(2700,remaining));save(folder/'resources.json',resource)
        p=folder/'result.json'
        if resource['returncode'] or not p.exists(): state['failures'].append(dict(job=label,resource=resource))
        elif json.loads(p.read_text())['status']=='excluded':state['exclusions'].append(dict(job=label,reason=json.loads(p.read_text()).get('reason')))
        state['jobs_done']=i+1;state['elapsed_seconds']=time.time()-started;save(a.out/'status.json',state);summarize(a.out)
        if len(state['failures'])>=3:state['status']='failed';break
    if state['jobs_done']==len(jobs):state.update(status='complete' if not state['failures'] else 'complete_with_failures',complete=not state['failures'])
    state['elapsed_seconds']=time.time()-started;save(a.out/'status.json',state);summarize(a.out)
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--bin',type=Path,required=True)
    p.add_argument('--worker',choices=['native','proof']);p.add_argument('--dataset',default='gaussian_hard');p.add_argument('--k',type=int,default=5)
    p.add_argument('--start',type=int,default=0);p.add_argument('--variant',default='protected_linear');p.add_argument('--budget',type=int,default=16)
    p.add_argument('--large',action='store_true');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();a.out=a.out.resolve();a.source=a.source.resolve();a.bin=a.bin.resolve();a.out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=4):
        if a.worker=='native':native_worker(a)
        elif a.worker=='proof':proof_worker(a)
        elif not campaign(a)['complete']:raise SystemExit(2)


if __name__=='__main__':main()
