"""Export completed review evidence after the serial queue, preserving authored text.

This read-only observer never launches proofs. Its final PDF is an unreviewed
candidate, distinct from the PDF inspected during the interactive session.
"""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
from campaigns.local_fast_tradeoff import save,sha

REPO=Path(__file__).resolve().parents[2]
PIN=REPO/'paper_results/referee_final_20261025/authoring_pin.json'


def read(path):return json.loads(path.read_text())


def main(root):
    status=root/'paper_finalizer.json'
    if status.exists():raise RuntimeError('Finalizer already has a receipt; inspect before retry')
    start=time.time();awake=False
    if os.name=='nt':
        import ctypes
        awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    save(status,dict(status='waiting',pid=os.getpid(),started_unix=start,prevent_idle_sleep=awake,
        permanent_power_settings_changed=False))
    try:
        while True:
            if (root/'STOP').exists():
                save(status,dict(status='user_stop_no_export',pid=None));return
            main=read(root/'queue.json');follow=read(root/'followups/queue.json')
            if main.get('complete') and follow.get('status') in ['complete','complete_with_retained_exclusions','stopped']:break
            if main.get('status') in ['stopped','interrupted']:break
            if time.time()>main['budget_started_unix']+26*3600:raise RuntimeError('Finalizer observation budget ended')
            time.sleep(30)
        ready=root/'reviewed_authoring_ready.json'
        if not ready.exists() or read(ready)['authoring_pin_sha256']!=sha(PIN):
            raise RuntimeError('No matching reviewed authoring pin; preserve manual edits')
        expected=read(PIN)['authored_sha256']
        if any(sha(REPO/name)!=digest for name,digest in expected.items()):
            raise RuntimeError('Authored manuscript changed; automatic update needs manual rebase')
        env=dict(os.environ,PYTHONPATH=str(REPO/'code'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        phases=[('old_completed_export','analysis.update_optimized_evidence',[]),
                ('new_completed_export','analysis.export_two_review_evidence',['--root',str(root)]),
                ('live_status','analysis.two_review_campaign_status',[]),
                ('paper_candidate','analysis.release_referee_final_paper',['--reuse-checked-tables','--automatic','--tag','prototype_audit_two_reviews_20261027'])]
        completed=[]
        for label,module,args in phases:
            save(status,dict(status='exporting',active_phase=label,pid=os.getpid(),completed=completed))
            with (root/('finalizer_'+label+'.log')).open('xb') as log:
                result=subprocess.run([sys.executable,'-u','-m',module]+args,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if result.returncode:raise RuntimeError(label+' failed; inspect retained log')
            completed.append(label)
        save(status,dict(status='candidate_awaiting_visual_review',pid=None,completed=completed,finished_unix=time.time(),
            reviewed_pdf_preserved=True,all_experiments_complete=bool(main.get('complete') and follow.get('complete')),
            submission_ready=False,public_release=False))
    except BaseException as error:
        save(status,dict(status='stopped_needs_review',pid=None,error=str(error),finished_unix=time.time()));raise
    finally:
        if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root.resolve())
