"""Run the optimized certificate after screens; escalate scale only after success."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,copy
import numpy as np
from campaigns.local_fast_tradeoff import save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','parent','binary','smoke-receipt','precision-screen','larger-root']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args();root=a.root.resolve();queue=root.with_name(root.name+'_queue.json')
    state=dict(complete=False,status='waiting_for_precision_screen',pid=os.getpid(),started_unix=time.time(),proof_root=str(root),larger_root=str(a.larger_root.resolve()),binary_sha256=sha(a.binary),large_gate='Fresh verification, no wrong decisions, frozen prediction accuracy>=99% on original matched workload, and measured Full ratio>=4. This is a resource-allocation goal, not a publication rule. Preserve all screens including rejected precisions.');save(queue,state)
    def invoke(command,phase):
        while True:
            child=subprocess.Popen(command,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));state.update(status='running',phase=phase,child_pid=child.pid);save(queue,state);code=child.wait()
            if code==0:return
            target=root if phase=='matched_control' else a.larger_root.resolve()/'compact'
            path=target/'status.json'
            if path.exists():
                data=json.loads(path.read_text());active=data.get('active_phase');attempts=data.get('resources',{}).get(active,{}).get('attempts',[])
                if active=='prospective_full_fallback':
                    fallback=target/'fallback';fallback_state=json.loads((fallback/'status.json').read_text());fallback_phase=fallback_state.get('active_phase');record=fallback/'full'/(str(fallback_phase)+'.resource.json');attempts=json.loads(record.read_text()).get('attempts',[]) if record.exists() else []
                    if fallback_state.get('paused'):data['paused']=True
            else:
                baseline=a.larger_root.resolve()/'parent';data=json.loads((baseline/'status.json').read_text());active=data.get('active_phase');record=baseline/'full'/(str(active)+'.resource.json');attempts=json.loads(record.read_text()).get('attempts',[]) if record.exists() else []
            if data.get('paused') or not attempts or attempts[-1].get('stop')!='RAM reserve':raise RuntimeError(f'{phase} stopped; inspect preserved outputs')
            state.update(status='waiting_for_ram_retry',child_pid=None);save(queue,state)
            if '--resume' not in command:command.append('--resume')
            time.sleep(15)
    try:
        while True:
            screen=json.loads((a.precision_screen/'status.json').read_text())
            if screen.get('complete'):break
            if screen.get('status')=='stopped':raise RuntimeError('Precision screen stopped; inspect logs before selecting a candidate')
            time.sleep(15)
        bits=screen['selected_bits'];state.update(selected_certificate_bits=bits);save(queue,state)
        smoke=json.loads(a.smoke_receipt.read_text());assert smoke['complete'] and smoke.get('compact_profile') and smoke['binary_sha256']==sha(a.binary)
        command=[sys.executable,'-m','campaigns.run_group_fast_control','--compact','--certificate-scale-bits',str(bits)]
        for name in ['root','parent','binary','smoke_receipt']:command+=['--'+name.replace('_','-'),str(getattr(a,name).resolve())]
        if a.resume:command.append('--resume')
        invoke(command,'matched_control');comparison=json.loads((root/'checked_comparison.json').read_text());assert comparison['complete'] and comparison['no_wrong_decisions'] and comparison['all_intervals_contain_full']
        repo=Path(__file__).resolve().parents[2];baseline=json.loads((repo/'paper_results/review_controls/completed_fast_baseline_20261024/summary.json').read_text());frozen=next(r for r in baseline['cases'] if r['case']=='head_r63');assert frozen['input_sha256']==sha(a.parent/'head_r63/input.json');assert frozen['minimum_agreement_percent']>=99.
        state['matched_result']=comparison
        subprocess.run([sys.executable,'-m','analysis.summarize_compact_optimization','--root',str(root),'--out',str(repo/'paper_results/review_controls/compact_optimization_20261025')],check=True)
        if comparison['full_over_group_active_phase']>=4.:
            state.update(status='building_larger_control',phase='larger');save(queue,state)
            command=[sys.executable,'-m','campaigns.run_compact_larger_control','--root',str(a.larger_root.resolve()),'--binary',str(a.binary.resolve()),'--smoke-receipt',str(a.smoke_receipt.resolve()),'--certificate-scale-bits',str(bits)]
            invoke(command,'larger_control');state.update(larger_control_ran=True)
        else:state.update(larger_control_ran=False,larger_deferred_reason='Measured saving did not reach the predefined4-fold resource gate. Preserve the result; optimize further before a larger campaign.')
        state.update(complete=True,status='complete',child_pid=None,finished_unix=time.time());save(queue,state)
    except BaseException as error:state.update(status='stopped',error=str(error));save(queue,state);raise


if __name__=='__main__':main()
