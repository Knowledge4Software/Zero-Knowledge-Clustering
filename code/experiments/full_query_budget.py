"""Two predeclared Full baselines just above the previous constraint ceiling."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--bin',type=Path,required=True);p.add_argument('--frozen-src',type=Path,required=True)
    p.add_argument('--gate',type=Path);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    state=dict(status='waiting' if a.gate else 'starting',complete=False,started_unix=time.time(),pid=os.getpid(),cases=[])
    def save_state():
        temp=a.out/'status.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(a.out/'status.json')
    save_state()
    try:
        if a.gate:
            deadline=time.monotonic()+7*3600
            while True:
                s=json.loads(a.gate.read_text()) if a.gate.exists() else {}
                if s.get('status') in ['complete','failed']:state['prior_status']=s['status'];break
                if time.monotonic()>deadline:raise RuntimeError('finite seven-hour queue gate')
                time.sleep(10)
        sys.path.insert(0,str(a.frozen_src.resolve()))
        import numpy as np
        import pandas as pd
        from experiments.paper_supplement import public_export
        from experiments.paper_overnight import run_bounded
        state.update(status='running',started_control_unix=time.time());save_state()
        rows=[]
        names=['synthetic_smoke'] if a.smoke else ['mnist','fashion_mnist']
        for name in names:
            if time.time()-state['started_control_unix']>2*3600:raise RuntimeError('two-hour control deadline')
            source=a.source/'cases'/f'{name}_k5_full';out=a.out/name;out.mkdir()
            if not json.loads((source/'verify.json').read_text()).get('complete'):raise RuntimeError('unverified source')
            bundle=out/'bundle';shutil.copytree(source,bundle)
            inp=json.loads((source/'input.json').read_text());q=4 if a.smoke else 8
            queries=[v for v in inp['queries'] if len(v)==q][:2]
            if not queries:raise RuntimeError('no predeclared matching source subsets')
            inp['queries']=queries;input_path=out/'private_input.json';input_path.write_text(json.dumps(inp))
            events=[]
            for phase,target in [('query',bundle),('verify',bundle),('verify',out/'public_bundle')]:
                if target.name=='public_bundle':public_export(bundle,target)
                cmd=[str(a.bin),'-phase',phase,'-instance',str(input_path),'-out',str(target),'-q',str(q),
                    '-max-constraints','1500000','-reps','1' if a.smoke else '3','-warmup','0' if a.smoke else '1']
                if phase=='verify':cmd+=['-trust',str(target/'trusted.json')]
                label=phase+('_public' if target.name=='public_bundle' else '')
                remaining=7200-(time.time()-state['started_control_unix'])
                if remaining<=0:raise RuntimeError('two-hour control deadline')
                resource=run_bounded(cmd,out/(label+'.log'),seconds=min(2400,remaining))
                path=target/(f'query_{q}.json' if phase=='query' else 'verify.json')
                result=json.loads(path.read_text()) if path.exists() else {}
                events.append(dict(phase=label,resource=resource,result=result))
                if resource['returncode'] not in [0,3] or resource['stop']:raise RuntimeError(f'{name}/{label}: {resource}')
                if not result.get('complete'):break
            query=events[0]['result'];verified=events[-1]['phase']=='verify_public' and events[-1]['result'].get('complete',False)
            samples=query.get('samples') or [];times=[np.median(s['prove_seconds']) for s in samples]
            row=dict(dataset=name,k=5,n_catalog=inp['model']['n'],q=q,status=query.get('status'),complete=verified,
                constraints=query.get('constraints',0),compile_seconds=query.get('compile_seconds',0),setup_seconds=query.get('setup_seconds',0),
                prove_median_seconds=float(np.median(times)) if times else None,
                peak_rss_bytes=max(e['resource']['peak_rss_bytes'] for e in events),independent_public_verification=verified)
            rows.append(row);pd.DataFrame(rows).to_csv(a.out/'full_q8_controls.csv',index=False)
            (out/'result.json').write_text(json.dumps(dict(summary=row,events=events),indent=2))
            state['cases'].append(row);save_state()
        state.update(status='complete',complete=all(r['complete'] for r in rows),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (a.out/'report.md').write_text('# Additional Full query-budget control\n\n'
            'Two fixed cases (MNIST/Fashion k=5, q=8), two existing distinct subsets each, one warmup and three measured repetitions. '
            'Only the constraint ceiling changes from 1.3M to 1.5M; the 5.3 GiB own-tree guard and host reserve remain active. '
            'Excluded/resource-limited instances are never successful measurements. Full needs no preprocessing proof. '
            'Public exports exclude private witnesses and assignment logs, and are independently verified.\n')
    except BaseException as e:
        state.update(status='failed',error=f'{type(e).__name__}: {e}');save_state();raise
    finally:
        state['elapsed_seconds']=time.time()-state['started_unix'];save_state()


if __name__=='__main__':main()
