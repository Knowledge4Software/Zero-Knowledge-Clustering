"""Compare old absolute-margin and tighter relative-margin label certificates."""
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from commitjl.relative_certificates import relative_certificate
from commitjl.reuse_certificates import certificate


def main():
    p=argparse.ArgumentParser();p.add_argument('--screen',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);records=[];start=time.monotonic()
    with threadpool_limits(limits=2):
        for file in sorted(a.screen.glob('*.npz')):
            z=np.load(file);audit=z['audit'];sets=z['centers'];basis=z['basis'];parts=file.stem.split('_')
            for rank in [15,31,63]:
                model=dict(basis=basis[:rank].tolist(),basis_scale=16384,centers=sets[0].tolist(),input_bits=12)
                for s in range(1,5):
                    centers=sets[s];old=certificate(model,audit,centers);new=relative_certificate(model,audit,centers)
                    assert old['labels']==new['labels']
                    truth=((audit[:,None,:]-centers[None,:,:])**2).sum(2).argmin(1)
                    assert all(not flag or label==t for flag,label,t in zip(new['certified_flags'],new['labels'],truth))
                    # Stronger rounding correction can differ; record rather than assume dominance.
                    records.append(dict(fit=file.stem,rank=rank,set=s,agreement_percent=100*np.mean(truth==new['labels']),
                        old_coverage_percent=100*old['certified_count']/len(audit),relative_coverage_percent=100*new['certified_count']/len(audit)))
            pd.DataFrame(records).to_csv(a.out/'results.csv',index=False)
            print(json.dumps(dict(completed=file.stem,elapsed=time.monotonic()-start)),flush=True)
    d=pd.DataFrame(records);d.groupby('rank')[['agreement_percent','old_coverage_percent','relative_coverage_percent']].agg(['mean','min']).to_csv(a.out/'groups.csv')
    (a.out/'summary.json').write_text(json.dumps(dict(complete=True,cells=len(d),false_certified_labels=0,elapsed=time.monotonic()-start,
        scope='Native assignment certificates only; not an exact cost-change decision or a proof speed claim.'),indent=2))
    print(d.groupby('rank')[['agreement_percent','old_coverage_percent','relative_coverage_percent']].agg(['mean','min']).to_string(),flush=True)


if __name__=='__main__':main()
