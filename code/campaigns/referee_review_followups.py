"""Serial follow-ups within the same24hour laptop budget, after the live queue."""
from pathlib import Path
import argparse,copy,json,os,time
import psutil
from campaigns.local_fast_tradeoff import save,sha
from campaigns.referee_two_reviews_local import Runner,REPO,OLD


def read(path):return json.loads(Path(path).read_text())


def run(root):
    follow=root/'followups';follow.mkdir(exist_ok=False)
    plan=read(root/'plan.json');save(follow/'plan.json',plan)
    parent=read(root/'queue.json')
    state=dict(complete=False,status='waiting_for_serial_prerequisite',started_unix=parent['started_unix'],
        budget_started_unix=parent.get('budget_started_unix',parent['started_unix']),completed=[],attempts=[],plan_sha256=sha(follow/'plan.json'))
    save(follow/'queue.json',state)
    while True:
        parent=read(root/'queue.json')
        if (root/'STOP').exists():raise RuntimeError('User stop file present')
        if parent.get('complete'):break
        if parent.get('status') in ['stopped','interrupted']:raise RuntimeError('Current queue needs explicit repair before follow-ups')
        if time.time()-state['budget_started_unix']>24*3600:raise RuntimeError('Declared campaign budget expired')
        time.sleep(5)
    runner=Runner(follow)
    try:
        # Every new neural activity outcome remains available, positive or not.
        runner.invoke('neural_activity_native',runner.py+['experiments.neural_activity_prototypes','--root',str(root)],limit=1800,resource_exclusion=True)
        inputs=root/'neural_activity_inputs/har_neural256_k6_seed307_classmean'
        if inputs.exists():runner.learning_proofs(inputs,root/'proof_har_neural256')
        selected=root/'backend_matched_inputs';full=selected/'ordered_head/full/input.json';cache=root/'unsafe_test_srs_cache'
        replacement=root/'backend/repetition_04'
        if (root/'backend_timing_interruption_amendment.json').exists():
            if runner.invoke('backend_replacement04',[str(runner.plonk),'-phase','run','-input',str(full),'-out',str(replacement),'-q','1','-allow-benchmark-srs','-srs-cache',str(cache)],limit=7200,resource_exclusion=True):
                runner.invoke('backend_replacement04_verify',[str(runner.plonk),'-phase','verify','-out',str(replacement),'-trust',str(replacement/'owner_pin.json'),'-public-task',str(replacement/'approved_task.json')],limit=600)
        # Match batch1 as well as the main batch8 Groth16 baseline.
        for rep in range(1,4):
            parent=follow/f'groth_batch1_r{rep:02d}';inp=read(full);inp['queries']=[[i] for i in range(inp['model']['n'])]
            save(parent/'full/input.json',inp);save(parent/'protocol.json',read(selected/'ordered_head/protocol.json'))
            runner.invoke(f'groth_batch1_r{rep}',runner.py+['campaigns.local_fast_tradeoff','--root',str(parent),'--binary',str(runner.binary),'--run','--cases','full'],limit=7200,resource_exclusion=True)
        # Try larger PLONK batches by feasibility, never by favorable timings.
        trials=[]
        for batch in [8,4,2]:
            inspect=follow/f'plonk_batch{batch}_inspect'
            if not runner.invoke(f'plonk_batch{batch}_inspect',[str(runner.plonk),'-phase','inspect','-input',str(full),'-out',str(inspect),'-q',str(batch)],limit=300,resource_exclusion=True):continue
            inspected=read(inspect/'compile_only.json')
            if not inspected['within_guard']:
                trials.append(dict(batch=batch,status='constraint_guard',constraints=inspected['constraints']));continue
            out=follow/f'plonk_batch{batch}'
            if runner.invoke(f'plonk_batch{batch}',[str(runner.plonk),'-phase','run','-input',str(full),'-out',str(out),'-q',str(batch),'-allow-benchmark-srs','-srs-cache',str(cache)],limit=7200,resource_exclusion=True):
                runner.invoke(f'plonk_batch{batch}_verify',[str(runner.plonk),'-phase','verify','-out',str(out),'-trust',str(out/'owner_pin.json'),'-public-task',str(out/'approved_task.json')],limit=600)
                trials.append(dict(batch=batch,status='complete',constraints=inspected['constraints']));break
            trials.append(dict(batch=batch,status='resource_guard',constraints=inspected['constraints']))
        save(follow/'plonk_batch_trials.json',trials)
        # Explicitly authorized continuation of the current interrupted affine
        # control only, not automatic revival of older research queues.
        parent=OLD/'extended/fashion_mnist_k5_seed173_age8_n1024/repetition_01/ordered_head'
        affine=parent.parent/'online_affine'
        runner.invoke('finish_current1024_affine',runner.py+['campaigns.run_online_affine_control','--parent',str(parent),'--root',str(affine),'--binary',str(runner.binary),'--run-proofs'],limit=4*3600,resource_exclusion=True)
        runner.invoke('export_current_controls',runner.py+['analysis.update_optimized_evidence','--root',str(OLD)],limit=600)
        runner.invoke('export_review_evidence',runner.py+['analysis.export_two_review_evidence','--root',str(root)],limit=600)
        runner.state.update(complete=True,status='complete_with_retained_exclusions' if any(a.get('stop') for a in runner.state['attempts']) else 'complete',finished_unix=time.time(),child_pid=None)
        save(follow/'queue.json',runner.state)
    except BaseException as e:
        runner.state.update(status='stopped',error=str(e),child_pid=None);save(follow/'queue.json',runner.state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root.resolve())
