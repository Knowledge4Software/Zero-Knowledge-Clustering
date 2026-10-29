"""Fresh matched auxiliary-rank proof cost after a successful native screen."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
import numpy as np
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits
from campaigns.local_fast_tradeoff import save,sha
from experiments.review_update_decisions import array_sha
from data.loaders import load_dataset


def prepare_basis(parent,path,rank):
    started=time.monotonic();inp=json.loads((parent/'head_r63/input.json').read_text());protocol=json.loads((parent/'protocol.json').read_text());old=np.asarray(inp['model']['centers'],dtype=np.int64);basis=np.asarray(inp['model']['basis'],dtype=np.int64)
    assert basis.shape==(63,784) and inp['model']['basis_scale']==16384
    with threadpool_limits(limits=1):
        x=np.asarray(load_dataset(protocol['dataset'],n_max=5000,seed=0)['X'],float);x=x[np.linalg.norm(x,axis=1)>0];indices=np.random.default_rng(741+protocol['fit_seed']).permutation(len(x));cut=int(.6*len(x));tr=x[indices[:cut]];scale=4095/max(float(np.abs(tr).max()),1e-12);training=np.rint(tr*scale).astype(np.int64);audit=np.clip(np.rint(x[indices[cut:cut+1024]]*scale),-4096,4096).astype(np.int64)
        assert array_sha(training)==protocol['training_sha256'] and array_sha(audit)==protocol['audit_sha256'];assert audit[:len(inp['x'])].tolist()==inp['x']
        q=np.linalg.qr(basis.T.astype(float),mode='reduced')[0];w=training.astype(float)-old[0];residual=w-(w@q)@q.T
        _,_,directions=randomized_svd(residual,n_components=64,n_iter=4,random_state=20261025);extra=np.rint(directions[:rank-63]*16384).astype(np.int64);combined=np.concatenate((basis,extra));assert np.array_equal(combined[:63],basis)
    result=dict(source_input_sha256=sha(parent/'head_r63/input.json'),training_sha256=array_sha(training),audit_sha256=array_sha(audit),basis_scale=16384,certificate_basis=combined.tolist(),rank=rank,fitting_seconds=time.monotonic()-started,recipe='Training-only residual randomized SVD with64 components,4 iterations,seed20261025, after QR projection out of the unchanged rounded original63 rows. Same recipe as the complete160-request screen.',selection='Same frozen N128 model/catalog and all19 distinct updates as the original matched control. No outcome-based update selection; no refitting of prediction model.')
    save(path,result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','parent','binary','smoke-receipt','native-screen']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--rank',type=int,choices=[79,95],default=79);p.add_argument('--resume',action='store_true');a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=a.resume);queue=root/'queue.json';state=dict(complete=False,status='preparing_auxiliary_basis',pid=os.getpid(),started_unix=time.time(),rank=a.rank,proof_root=str(root/'control'),binary_sha256=sha(a.binary),scope='Fresh matched N128 cost-control only. Higher-rank160-request sensitivity remains a separate native result until independently proved. Same precision14, q16, original prediction model and all19 updates. No joint speed/sensitivity replication claim.');save(queue,state)
    try:
        screen=json.loads((a.native_screen/'summary.json').read_text());assert screen['complete'] and screen['false_conclusive']==0 and screen['all_intervals_contain_truth'];result=screen['results_by_rank'][str(a.rank)];assert result['requests']==160 and result['conclusive']>=151 and result['detected']>=34
        state['native_screen_sha256']=sha(a.native_screen/'summary.json');basis=root/'auxiliary_basis.json'
        if not basis.exists():prepare_basis(a.parent.resolve(),basis,a.rank)
        else:assert json.loads(basis.read_text())['rank']==a.rank
        command=[sys.executable,'-m','campaigns.run_group_fast_control','--compact','--certificate-scale-bits','14','--auxiliary-basis',str(basis),'--root',str(root/'control'),'--parent',str(a.parent.resolve()),'--binary',str(a.binary.resolve()),'--smoke-receipt',str(a.smoke_receipt.resolve())]
        if a.resume:command.append('--resume')
        while True:
            child=subprocess.Popen(command,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));state.update(status='running',child_pid=child.pid);save(queue,state);code=child.wait()
            if code==0:break
            status=json.loads((root/'control/status.json').read_text());phase=status.get('active_phase');attempts=status.get('resources',{}).get(phase,{}).get('attempts',[])
            if phase=='prospective_full_fallback':
                fallback=root/'control/fallback';fs=json.loads((fallback/'status.json').read_text());record=fallback/'full'/(str(fs['active_phase'])+'.resource.json');attempts=json.loads(record.read_text()).get('attempts',[]) if record.exists() else []
                if fs.get('paused'):status['paused']=True
            if status.get('paused') or not attempts or attempts[-1].get('stop')!='RAM reserve':raise RuntimeError('Proof worker stopped; inspect preserved log and charged attempts')
            state.update(status='waiting_for_ram_retry',child_pid=None);save(queue,state)
            if '--resume' not in command:command.append('--resume')
            time.sleep(15)
        result=json.loads((root/'control/checked_comparison.json').read_text());assert result['complete'] and result['no_wrong_decisions'] and result['all_intervals_contain_full']
        repo=Path(__file__).resolve().parents[2];subprocess.run([sys.executable,'-m','analysis.summarize_compact_optimization','--root',str(root/'control'),'--out',str(repo/'paper_results/review_controls'/f'compact_auxiliary_rank{a.rank}_20261025')],check=True)
        state.update(complete=True,status='complete',child_pid=None,finished_unix=time.time(),result=result);save(queue,state)
    except BaseException as error:state.update(status='stopped',error=str(error));save(queue,state);raise


if __name__=='__main__':main()
