"""Screen one intermediate rank on an existing, unchanged audit workload."""
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from campaigns.local_fast_tradeoff import save,sha
from commitjl.relative_certificates import relative_certificate
from commitjl.reuse_certificates import certificate
from experiments.review_update_decisions import array_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--rank',type=int,required=True);a=p.parse_args()
    parent=a.parent.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    source=parent/'head_r63/input.json';inp=json.loads(source.read_text());protocol=json.loads((parent/'protocol.json').read_text())
    if not 1<=a.rank<=len(inp['model']['basis']):raise ValueError('rank must be a prefix of the frozen source basis')
    started=time.monotonic();archive=np.load(a.archive);audit=archive['audit']
    assert array_sha(audit)==protocol['audit_sha256'] and np.array_equal(audit[:len(inp['x'])],inp['x'])
    assert np.array_equal(archive['basis'],inp['model']['basis'])
    inp['model']['basis']=inp['model']['basis'][:a.rank];save(out/'input.json',inp)
    records=[]
    with threadpool_limits(limits=1):
        for scope,rows in [('native1024',audit),('proof_catalog',np.asarray(inp['x'],dtype=np.int64))]:
            for s,centers in enumerate(inp['reuse_centers']):
                absolute=certificate(inp['model'],rows,centers);relative=relative_certificate(inp['model'],rows,centers)
                distances=((rows[:,None,:]-np.asarray(centers)[None,:,:])**2).sum(2);labels=distances.argmin(1);truth=int(distances.min(1).sum())
                assert absolute['labels']==relative['labels']
                assert all(not flag or lab==full for flag,lab,full in zip(relative['certified_flags'],relative['labels'],labels))
                assert absolute['cost_lower_scaled']<=truth*absolute['cost_denominator']<=absolute['cost_upper_scaled']
                records.append(dict(rank=a.rank,scope=scope,set=s,lloyd_step=protocol['lloyd_steps'][s],n=len(rows),
                    agreement_percent=100*np.mean(labels==relative['labels']),relative_coverage_percent=100*relative['certified_count']/len(rows),
                    absolute_coverage_percent=100*absolute['certified_count']/len(rows),
                    absolute_cost_error_percent=100*abs(absolute['psi']/inp['model']['basis_scale']**4/truth-1),
                    cost_interval_width_percent=100*(absolute['cost_upper_scaled']-absolute['cost_lower_scaled'])/absolute['cost_denominator']/truth))
    frame=pd.DataFrame(records);frame.to_csv(out/'quality.csv',index=False)
    groups=frame.groupby('scope').agg(sets=('set','count'),agreement_mean=('agreement_percent','mean'),agreement_min=('agreement_percent','min'),
        relative_coverage_mean=('relative_coverage_percent','mean'),relative_coverage_min=('relative_coverage_percent','min'),
        old_coverage_mean=('absolute_coverage_percent','mean'),cost_error_max=('absolute_cost_error_percent','max'),width_max=('cost_interval_width_percent','max')).reset_index()
    groups['passes_user_99_on_every_set']=groups.agreement_min>=99;groups.to_csv(out/'groups.csv',index=False)
    save(out/'summary.json',dict(complete=True,rank=a.rank,elapsed_seconds=time.monotonic()-started,scope='Native numerical screen; no measured proof time for this rank.',
        source_input_sha256=sha(source),archive_sha256=sha(a.archive),all_center_sets_retained=True,original_ordered_basis_prefix=True,
        false_certified_labels=0,all_absolute_intervals_contain_full_cost=True,groups=groups.to_dict('records')))
    print(groups.to_string(index=False),flush=True)


if __name__=='__main__':main()
