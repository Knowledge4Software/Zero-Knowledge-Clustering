"""Generate an extended held-out catalog and check integer certificates locally.

Reuses the published N=256 bases, centers and quantizer. No refitting or audit-row
training. The original 256 quantized rows must exactly match before extending.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import numpy as np
import pandas as pd

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from commitjl.reuse_certificates import summarize
from data.loaders import load_dataset
from experiments.bound_audit import save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--references',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--n',type=int,default=1024);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    learned=json.loads((a.references/'reuse_mnist_n256_learned_energy.json').read_text())
    data=load_dataset('mnist',n_max=5000,seed=0);x=np.asarray(data['X'],float)
    x=x[np.linalg.norm(x,axis=1)>0]
    order=np.random.default_rng(10000).permutation(len(x));audit=order[int(.6*len(x)):]
    if a.n>len(audit) or a.n&(a.n-1):raise ValueError('need a power-of-two actual held-out catalog within loaded data')
    xi=np.rint(x[audit[:a.n]]*float(learned['model']['input_scale'])).astype(np.int64)
    if not np.array_equal(xi[:256],np.asarray(learned['x'])):raise AssertionError('local data differs from the original owner catalog')
    new_inputs=[]
    for method in ['learned_energy','pca_energy']:
        source=a.references/f'reuse_mnist_n256_{method}.json';inp=json.loads(source.read_text())
        if inp['reuse_centers']!=learned['reuse_centers'] or inp['x']!=learned['x'] or inp['salt_x']!=learned['salt_x']:
            raise AssertionError('unmatched source catalogs')
        inp=copy.deepcopy(inp);inp['model']['n']=a.n;inp['x']=xi.tolist()
        inp['queries']=[list(range(s,s+64)) for s in range(0,a.n,64)]
        path=a.out/'owner_inputs'/f'mnist_n{a.n}_{method}.json';save(path,inp);new_inputs.append(path)
    rows=[]
    sources=sorted(a.references.glob('reuse_*.json'))+new_inputs
    for source in sources:
        inp=json.loads(source.read_text());method='pca_energy' if 'pca_energy' in source.name else 'learned_energy'
        dataset='cifar10_resnet50' if 'cifar' in source.name else 'mnist'
        for s,centers in enumerate(inp['reuse_centers']):
            row=summarize(inp['model'],inp['x'],centers)
            row.update(dataset=dataset,method=method,set=s,mode='native_preflight',source_input=str(source))
            rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(a.out/'certificate_preflight.csv',index=False)
    summary=frame[frame['set']>0].groupby(['dataset','n','method'],as_index=False).agg(
        mean_agreement=('agreement_quantized','mean'),mean_certified_fraction=('certified_fraction','mean'),
        min_certified_fraction=('certified_fraction','min'),mean_interval_relative_width=('interval_relative_width','mean'),
        exact_label_policy_sets=('exact_label_policy_accept','sum'),coverage99_cost1_policy_sets=('coverage99_cost1_policy_accept','sum'))
    summary.to_csv(a.out/'new_set_summary.csv',index=False)
    save(a.out/'manifest.json',dict(n=a.n,prefix_matches_original_256=True,training_refit=False,
        certificate_target='committed quantized records; native preflight, not proof receipts',
        sources=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sources],
        data_metadata=data.get('meta',{}),certified_wrong_rows=int(frame.certified_wrong_rows.sum()),
        intervals_contain_reference=bool(frame.interval_contains_exact_cost.all())))
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':main()
