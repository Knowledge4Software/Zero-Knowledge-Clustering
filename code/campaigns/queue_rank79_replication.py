"""Serial laptop replication: new-fit screen, fresh Full and compact proofs.

No quality gate removes a fit/update. Every inconclusive public decision gets
newly generated Full fallback proofs, whose costs remain charged.
"""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
import psutil
from commitjl.pilot_io import save,sha

REPO=Path(__file__).resolve().parents[2]
PROOF_FITS=['fashion_mnist_k3_seed101_age8','mnist_k5_seed113_age1']


def read(path):return json.loads(path.read_text())


def adopt_saved_child(state,phase,command):
    pid=state.get('child_pid')
    if not pid:return None
    try:child=psutil.Process(pid)
    except psutil.NoSuchProcess:return None
    if state.get('phase')!=phase or child.cmdline()!=command:
        raise RuntimeError('Saved live child identity differs; refusing a duplicate campaign worker')
    if state.get('child_created_unix') is not None and abs(child.create_time()-state['child_created_unix'])>1:
        raise RuntimeError('Saved child PID was reused; refusing to adopt it')
    return child


def completed_child_status(status_path,full_parent=None):
    if status_path is None or not status_path.exists():return False
    state=read(status_path)
    if not state.get('complete') or state.get('paused') or state.get('status')=='stopped':return False
    if full_parent is not None:return bool(state.get('cases',{}).get('full',{}).get('complete'))
    comparison=status_path.parent/'checked_comparison.json'
    return comparison.exists() and bool(read(comparison).get('complete'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['root','native-out','binary','smoke-receipt','wait-for-jl']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--wait-for-secondary-jl',type=Path)
    p.add_argument('--resume',action='store_true');a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=a.resume)
    state=read(root/'queue.json') if a.resume else dict(complete=False,started_unix=time.time(),binary_sha256=sha(a.binary),proof_fits=PROOF_FITS,scope='Two predeclared N128 fits, one repetition, head63/head79/PCA79 and equally reusable public-center Full. All distinct training updates, fresh cold setups, public-only replay, prospective generation of every fallback. Dataset/seed/k choices fixed before outcomes. Native1024 evidence reported separately.',completed=[])
    assert state['binary_sha256']==sha(a.binary);state.update(pid=os.getpid(),status='waiting_for_jl',paused=False);save(root/'queue.json',state)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def invoke(command,phase,status_path=None,full_parent=None):
        while True:
            # A coordinator may restart while its independently running worker
            # continues. Keep the exact original argv if only the coordinator
            # would otherwise add a new --resume flag on that same live worker.
            if state.get('child_pid') and psutil.pid_exists(state['child_pid']) and state.get('phase')==phase and state.get('child_command')==command[:-1] and command[-1:]==['--resume']:
                command=state['child_command']
            adopted=adopt_saved_child(state,phase,command)
            with (root/(phase+'.stdout.log')).open('a',encoding='utf-8') as out,(root/(phase+'.stderr.log')).open('a',encoding='utf-8') as err:
                if adopted is None:
                    child=subprocess.Popen(command,env=env,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    created=psutil.Process(child.pid).create_time()
                else:
                    child=adopted;created=child.create_time()
                    receipt=dict(phase=phase,pid=child.pid,command=command,child_created_unix=created,adopted_unix=time.time(),coordinator_pid=os.getpid(),scope='Coordinator adoption only; existing worker/proof attempt and its generation-cost records continue unchanged.')
                    state.setdefault('child_adoptions',[]).append(receipt);save(root/(phase+f'.adoption_{time.time_ns()}.json'),receipt)
                state.update(status='running',phase=phase,child_pid=child.pid,child_command=command,child_created_unix=created);save(root/'queue.json',state)
                try:code=child.wait()
                except psutil.NoSuchProcess:code=None
            if adopted is not None and code in (0,None):code=0 if completed_child_status(status_path,full_parent) else 1
            elif code==0 and status_path is not None and not completed_child_status(status_path,full_parent):code=1
            if code==0:
                if phase not in state['completed']:state['completed'].append(phase)
                state.update(child_pid=None);save(root/'queue.json',state);return
            if status_path is None:raise RuntimeError(f'{phase} failed; inspect preserved logs')
            data=read(status_path);active=data.get('active_phase');attempts=data.get('resources',{}).get(active,{}).get('attempts',[])
            if full_parent is not None:
                record=full_parent/'full'/(str(active)+'.resource.json');attempts=read(record).get('attempts',[]) if record.exists() else []
            elif active=='prospective_full_fallback':
                fb=status_path.parent/'fallback';fs=read(fb/'status.json');record=fb/'full'/(str(fs.get('active_phase'))+'.resource.json');attempts=read(record).get('attempts',[]) if record.exists() else []
                if fs.get('paused'):data['paused']=True
            if data.get('paused') or not attempts or attempts[-1].get('stop')!='RAM reserve':raise RuntimeError(f'{phase} stopped; inspect preserved attempts/logs')
            state.update(status='waiting_for_ram_retry',child_pid=None);save(root/'queue.json',state)
            if full_parent is None and '--resume' not in command:command.append('--resume')
            time.sleep(15)
    try:
        while not (a.wait_for_jl/'summary.json').exists():
            path=a.wait_for_jl/'status.json'
            if path.exists():
                js=read(path)
                if js.get('pid') and not psutil.pid_exists(js['pid']):raise RuntimeError('JL screen stopped; inspect its logs before continuing')
            time.sleep(15)
        assert read(a.wait_for_jl/'summary.json')['complete']
        if not (a.native_out/'summary.json').exists():
            command=[sys.executable,'-u','-m','experiments.rank79_independent_replication','--out',str(a.native_out.resolve())]
            adopted=None
            if a.native_out.exists():
                if not a.resume:raise RuntimeError('Partial native screen retained; explicit resume required, do not overwrite')
                previous=read(a.native_out/'status.json')
                if previous.get('pid') and psutil.pid_exists(previous['pid']):
                    proc=psutil.Process(previous['pid']);cmd=proc.cmdline()
                    assert 'experiments.rank79_independent_replication' in cmd and str(a.native_out.resolve()) in cmd,'live native identity mismatch'
                    adopted=proc;state.update(status='running',phase='native_replication',child_pid=proc.pid,adopted_native=True);save(root/'queue.json',state)
                command.append('--resume')
            if adopted is None:invoke(command,'native_replication')
            else:
                code=adopted.wait()
                if code not in (0,None) or not (a.native_out/'summary.json').exists():raise RuntimeError('Adopted native worker stopped; inspect preserved checkpoints')
        native=read(a.native_out/'summary.json');assert native['complete'] and native['false_conclusive']==0 and native['all_intervals_contain_truth']
        if a.wait_for_secondary_jl:
            while not (a.wait_for_secondary_jl/'summary.json').exists():
                state.update(status='waiting_for_secondary_jl',phase='native_replication',child_pid=None);save(root/'queue.json',state)
                path=a.wait_for_secondary_jl/'status.json'
                if path.exists():
                    previous=read(path)
                    if previous.get('pid') and not psutil.pid_exists(previous['pid']):raise RuntimeError('Secondary JL pilot stopped; inspect its logs')
                time.sleep(15)
            assert read(a.wait_for_secondary_jl/'summary.json')['complete']
        result=[]
        for fit in PROOF_FITS:
            parent=a.native_out.resolve()/'proof_inputs'/f'{fit}_ordered_head'
            if not (root/fit/'full_reference.json').exists():
                invoke([sys.executable,'-u','-m','campaigns.local_fast_tradeoff','--run','--root',str(parent),'--cases','full','--binary',str(a.binary.resolve())],fit+'_full',parent/'status.json',parent)
                from analysis.audit_local_fast_tradeoff import audit_case
                full,_,_=audit_case(parent,'full');save(root/fit/'full_reference.json',full)
            for name,representation,rank in [('head79','ordered_head',79),('head63','ordered_head',63),('pca79','pca',79)]:
                comparison=root/fit/name/'checked_comparison.json'
                if comparison.exists() and read(comparison).get('complete'):
                    result.append(dict(fit=fit,variant=name,**read(comparison)));continue
                source=a.native_out.resolve()/'proof_inputs'/f'{fit}_{representation}'
                # PCA uses the same exact public-center Full reference/input.
                if representation=='pca':
                    from campaigns.shared_full_reference import share_full_reference
                    share_full_reference(parent,source)
                command=[sys.executable,'-u','-m','campaigns.run_group_fast_control','--compact','--parent',str(source),'--root',str(comparison.parent),'--binary',str(a.binary.resolve()),'--smoke-receipt',str(a.smoke_receipt.resolve())]
                if rank==79:command+=['--auxiliary-basis',str(source/'auxiliary_basis.json')]
                if comparison.parent.exists():command.append('--resume')
                invoke(command,fit+'_'+name,comparison.parent/'status.json')
                r=read(comparison);assert r['complete'] and r['all_intervals_contain_full'] and r['no_wrong_decisions'];result.append(dict(fit=fit,variant=name,**r));save(root/'partial_comparison.json',result)
        state.update(complete=True,status='complete',child_pid=None,finished_unix=time.time());save(root/'summary.json',dict(complete=True,results=result,scope=state['scope'],native_summary_sha256=sha(a.native_out/'summary.json')));save(root/'queue.json',state)
    except BaseException as error:
        state.update(status='stopped',error=str(error));save(root/'queue.json',state);raise


if __name__=='__main__':main()
