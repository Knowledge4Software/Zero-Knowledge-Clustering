"""Predeclared local-center frontier and controlled nested-rank experiment.

This campaign uses already fitted training-only bases. It never chooses proof
instances from native outcomes. Integer certificates target quantized data.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from commitjl.reuse_certificates import certificate
from experiments.bound_audit import save

METHODS = ('learned_energy', 'pca_energy')
RANKS = (15, 63)
AMPLITUDES = (0.0001, 0.001, 0.01, 0.1, 0.3)
SEEDS = (17, 29)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def center_frontier(centers, input_bits, amplitudes=AMPLITUDES, seeds=SEEDS):
    """Directions and scale depend only on deployed public centers, never X."""
    c = np.asarray(centers, dtype=np.int64)
    if c.ndim != 2 or len(c) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('invalid centers or duplicate seeds')
    bound = 2**int(input_bits)
    if np.abs(c).max() > bound:
        raise ValueError('deployed centers outside circuit input range')
    cf = c.astype(float)
    pairwise = [np.linalg.norm(cf[i]-cf[j]) for i in range(len(c)) for j in range(i)]
    scale = float(np.median(pairwise))
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError('positive median center separation required')
    sets = [c.tolist()]
    metadata = [dict(set=0, seed=None, requested_relative_drift=0., actual_relative_drift=0.,
                     clipped_coordinates=0, unchanged_after_quantization=True)]
    directions = {}
    for seed in seeds:
        direction = np.random.default_rng(seed).normal(size=c.shape)
        direction /= np.sqrt(np.mean(np.sum(direction**2, axis=1)))
        directions[seed] = direction
    for amplitude in amplitudes:
        if not 0 < amplitude <= 1 or not np.isfinite(amplitude):
            raise ValueError('amplitudes must be finite in (0,1]')
        for seed in seeds:
            rounded = np.rint(cf + amplitude*scale*directions[seed])
            updated = np.clip(rounded, -bound, bound).astype(np.int64)
            drift = float(np.sqrt(np.mean(np.sum((updated.astype(float)-cf)**2, axis=1)))/scale)
            metadata.append(dict(set=len(sets), seed=int(seed), requested_relative_drift=amplitude,
                                 actual_relative_drift=drift,
                                 clipped_coordinates=int(np.count_nonzero(updated != rounded)),
                                 unchanged_after_quantization=bool(np.array_equal(updated,c))))
            sets.append(updated.tolist())
    return sets, metadata, scale


def nested_owner(source, rows, centers, rank, n, batch=16):
    if not 0 < rank <= len(source['model']['basis']) or n > len(rows) or n % batch:
        raise ValueError('invalid rank or catalog/batch size')
    result = copy.deepcopy(source)
    result['model']['basis'] = result['model']['basis'][:rank]
    result['model']['n'] = n
    result['x'] = copy.deepcopy(rows[:n])
    result['reuse_centers'] = copy.deepcopy(centers)
    result['queries'] = [list(range(i,i+batch)) for i in range(0,n,batch)]
    return result


def prepare(budget, catalog, out):
    # Refuse to overwrite any prior preparation, including incomplete attempts.
    out.mkdir(parents=True, exist_ok=False)
    sources = {}
    owners = {}
    for method in METHODS:
        b = budget/f'mnist_n256_{method}_r63_t16384.json'
        x = catalog/f'mnist_n1024_{method}.json'
        sources[method] = (json.loads(b.read_text()), json.loads(x.read_text()))
        owners[method] = dict(basis_source=str(b.resolve()), basis_sha256=sha(b),
                              catalog_source=str(x.resolve()), catalog_sha256=sha(x))
    reference, reference_large = sources[METHODS[0]]
    for method, (small, large) in sources.items():
        if len(small['model']['basis']) != 63 or small['model']['basis_scale'] != 16384:
            raise ValueError('requires frozen rank63/T16384 basis')
        for key in ('centers','input_scale','input_bits','d','k'):
            if small['model'][key] != large['model'][key] or small['model'][key] != reference['model'][key]:
                raise ValueError('incompatible frozen source '+key)
        if small['x'] != large['x'][:256] or large['x'] != reference_large['x']:
            raise ValueError('catalog prefixes differ')
        for key in ('salt_x','salt_z'):
            if small[key] != large[key] or small[key] != reference[key]:
                raise ValueError('owner salts differ')
    sets, metadata, scale = center_frontier(reference['model']['centers'], reference['model']['input_bits'])
    jobs = []
    inputs = []
    for method, (small, large) in sources.items():
        for rank in RANKS:
            for n in (256,1024):
                name = f'mnist_n{n}_{method}_nested_r{rank}_t16384'
                path = out/'owner_inputs'/(name+'.json')
                save(path,nested_owner(small,large['x'],sets,rank,n))
                inputs.append(dict(name=name,path=path.relative_to(out).as_posix(),sha256=sha(path),
                                   method=method,rank=rank,n=n))
                if n == 256:
                    jobs.append(dict(name=name,input=path.name,batch=16,full=False,certified=True))
    # Exactly the same quantized rows and all eleven center sets as the reuse cases.
    full_input = next(x['path'] for x in inputs if x['n']==256)
    jobs.append(dict(name='mnist_n256_full_public',input=Path(full_input).name,batch=8,full=True,certified=False))
    protocol = dict(version='local-center-frontier-v1',source_owners=owners,inputs=inputs,jobs=jobs,
        center_sets=metadata,median_public_center_separation=scale,
        ranks=list(RANKS),basis_scale=16384,reps=3,warmup=1,
        native_catalog_sizes=[256,1024],proof_catalog_size=256,
        selection='All predeclared sets proved, including refused and quantization-unchanged sets; no native outcome selection.',
        basis_policy='Nested prefixes of frozen training-only rank63 basis, rounded once at T=16384; no refitting.',
        drift_policy='RMS center displacement / median pairwise deployed-center distance; round then clip to circuit range.',
        strict_policy=dict(minimum_coverage=.99,maximum_relative_cost_width=.01),
        supplementary_policy=dict(minimum_coverage=.95,maximum_relative_cost_width=.05),
        timing_policy='Three repetitions per query proof, median for query totals; one timing per one-time Stage1 row-binding proof. Individual timings retained. Not independent dataset replicates.',
        limitation='Local update sensitivity on one fitted MNIST model; two direction seeds are not independent fitted-model replicates. Exact Full uses batch8, reuse uses batch16; compare each feasible configuration, not batch-isolated circuit speed.',
        status='predeclared; no campaign outcomes used')
    save(out/'protocol.json',protocol)
    return protocol


def native_summary(owner, centers):
    model = owner['model']
    x = np.asarray(owner['x'],dtype=np.int64)
    c = np.asarray(centers,dtype=np.int64)
    max_delta = int(np.abs(x).max())+int(np.abs(c).max())
    if max_delta**2*x.shape[1]*len(x) >= 2**63:
        raise OverflowError('exact distance reference exceeds int64 safety bound')
    distances = ((x[:,None,:]-c[None,:,:])**2).sum(axis=2)
    labels = distances.argmin(axis=1)
    cost = int(distances[np.arange(len(x)),labels].sum())
    cert = certificate(model,owner['x'],centers)
    wrong = sum(flag and a != b for flag,a,b in zip(cert['certified_flags'],cert['labels'],labels))
    lo,hi = cert['cost_lower_scaled'],cert['cost_upper_scaled']
    if wrong or not lo <= cost*cert['cost_denominator'] <= hi:
        raise AssertionError('native certificate contradicts exact quantized reference')
    count = cert['certified_count']
    return dict(agreement_quantized=float(np.mean(np.asarray(cert['labels'])==labels)),
        certified_count=count,certified_fraction=count/len(x),certified_wrong_rows=wrong,
        interval_contains_exact_cost=True,interval_relative_width=(hi-lo)/lo if lo else None,
        quantized_true_cost=cost,cost_lower_scaled=str(lo),cost_upper_scaled=str(hi),
        cost_denominator=str(cert['cost_denominator']),gram_defect=cert['gram_defect'],
        exact_label_policy_accept=count==len(x),
        coverage99_cost1_policy_accept=100*count>=99*len(x) and lo>0 and 100*(hi-lo)<=lo,
        coverage95_cost5_policy_accept=100*count>=95*len(x) and lo>0 and 20*(hi-lo)<=lo)


def run_native(prepared, out):
    out.mkdir(parents=True,exist_ok=False)
    protocol = json.loads((prepared/'protocol.json').read_text())
    rows = []
    for entry in protocol['inputs']:
        path = prepared/entry['path']
        if sha(path) != entry['sha256']:
            raise ValueError('prepared input hash changed')
        owner = json.loads(path.read_text())
        for metadata,centers in zip(protocol['center_sets'],owner['reuse_centers']):
            row = native_summary(owner,centers)
            row.update(method=entry['method'],rank=entry['rank'],n=entry['n'],**metadata)
            rows.append(row)
            print(json.dumps({k:row[k] for k in ('method','rank','n','set','certified_fraction','coverage99_cost1_policy_accept')}),flush=True)
        pd.DataFrame(rows).to_csv(out/'cells.csv',index=False)
    frame = pd.DataFrame(rows)
    updated = frame[frame['set']>0]
    grouped = updated.groupby(['method','rank','n','requested_relative_drift'],as_index=False).agg(
        directions=('set','count'),agreement=('agreement_quantized','mean'),
        certified_fraction=('certified_fraction','mean'),minimum_certified_fraction=('certified_fraction','min'),
        interval_relative_width=('interval_relative_width','mean'),
        strict_accepted_sets=('coverage99_cost1_policy_accept','sum'),
        supplementary_accepted_sets=('coverage95_cost5_policy_accept','sum'))
    grouped.to_csv(out/'frontier.csv',index=False)
    save(out/'summary.json',dict(complete=True,cells=len(rows),false_certified_rows=int(frame['certified_wrong_rows'].sum()),
        strict_accepted_updated_cells=int(updated['coverage99_cost1_policy_accept'].sum()),
        supplementary_accepted_updated_cells=int(updated['coverage95_cost5_policy_accept'].sum()),
        updated_cells=len(updated),protocol_sha256=sha(prepared/'protocol.json')))


def finalize(prepared, runs, out):
    """Require all cases/repetitions/public replays and native/proved agreement."""
    protocol = json.loads((prepared/'protocol.json').read_text())
    native = pd.read_csv(runs/'native/cells.csv',dtype={'cost_lower_scaled':str,'cost_upper_scaled':str})
    report = []
    for job in protocol['jobs']:
        root = runs/job['name']
        summary = json.loads((root/'summary.json').read_text())
        if not summary['complete'] or summary['sets'] != len(protocol['center_sets']):
            raise ValueError('incomplete proof case')
        worker_protocol = json.loads((root/'protocol.json').read_text())
        if worker_protocol['reps'] != 3 or worker_protocol['warmup'] != 1:
            raise ValueError('timing repetition policy mismatch')
        workload = pd.read_csv(root/'workload.csv',dtype={'cost_lower_scaled':str,'cost_upper_scaled':str})
        if job['certified']:
            entry = next(e for e in protocol['inputs'] if Path(e['path']).name == job['input'])
            ref = native[(native['method']==entry['method']) & (native['rank']==entry['rank']) & (native['n']==256)].set_index('set')
            for _,row in workload.iterrows():
                expected = ref.loc[row['set']]
                if (row['certificate_count'] != expected['certified_count'] or
                    int(row['cost_lower_scaled']) != int(expected['cost_lower_scaled']) or
                    int(row['cost_upper_scaled']) != int(expected['cost_upper_scaled'])):
                    raise ValueError('proved/native frontier mismatch')
        samples = []
        phases = ['full_reuse'] if job['full'] else ['stage1','reuse','reuse_certified']
        for phase in phases:
            filename = phase+'.json' if phase=='stage1' else f'{phase}_{job["batch"]}.json'
            data = json.loads((root/'bundle'/filename).read_text())
            for sample in data['samples']:
                values = sample['prove_seconds']
                expected = 1 if phase=='stage1' else 3
                if len(values) != expected or not all(v>0 for v in values):
                    if phase=='stage1':
                        raise ValueError('Stage1 requires one positive row-binding timing')
                    raise ValueError('missing three positive query proof timing repetitions')
                samples.append(dict(phase=phase,id=sample['id'],set=sample.get('centre_set'),
                    expected_repetitions=expected,prove_seconds=values,
                    median=float(np.median(values)),minimum=min(values),maximum=max(values)))
        if summary['verified_public_samples'] != len(samples):
            raise ValueError('public replay count differs from proof count')
        report.append(dict(name=job['name'],summary=summary,timing_samples=samples))
    if out.exists():
        raise FileExistsError(out)
    save(out,dict(complete=True,protocol_sha256=sha(prepared/'protocol.json'),jobs=report,
        timing_policy_clarification='The verifier --reps flag measures query proofs only. Stage1 row-binding proofs have one timing each; query proofs have three. No missing query repetitions are permitted.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','native','finalize'])
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--out',type=Path)
    p.add_argument('--budget',type=Path)
    p.add_argument('--catalog',type=Path)
    p.add_argument('--runs',type=Path)
    a = p.parse_args()
    if a.phase=='prepare': prepare(a.budget,a.catalog,a.prepared)
    elif a.phase=='native': run_native(a.prepared,a.out)
    else: finalize(a.prepared,a.runs,a.out)


if __name__=='__main__':
    main()
