"""Fresh public-only replay after the serial matched campaign completes.

This is an additional artifact audit using the same frozen verifier
implementations in new processes, not an independent security implementation.
Copies preserve the original timing/replay receipts. Audit time is separate
from measured deployment components.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,time

REPO=Path(__file__).resolve().parents[2]


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));os.replace(temp,path)


def replay_environment(workers=1,memory_mib=256):
    if type(workers) is not int or not 1<=workers<=8:raise ValueError('replay workers must be 1..8')
    if type(memory_mib) is not int or not 128<=memory_mib<=1200:raise ValueError('replay memory must be 128..1200 MiB')
    # The frozen moments verifier requires this accounting mode even though
    # verify-only phases never create keys or proofs.
    return dict(os.environ,COMMITJL_RESUME_PROOFS='1',GOMAXPROCS=str(workers),GOMEMLIMIT=f'{memory_mib}MiB',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')


def run(root,wait,audit_root=None,workers=1,memory_mib=256):
    audit=audit_root or root/'final_public_replay';audit.mkdir(parents=True,exist_ok=True)
    if (audit/'status.json').exists():raise ValueError('Use a fresh audit directory; preserve earlier replay receipts')
    env=replay_environment(workers,memory_mib)
    state=dict(complete=False,status='waiting_for_campaign',pid=os.getpid(),started_unix=time.time(),results=[],verification_only=True,workers=workers,go_soft_memory_mib=memory_mib)
    save(audit/'status.json',state)
    try:
        while not (root/'summary.json').exists() or not read(root/'summary.json').get('complete'):
            queue=read(root/'queue.json')
            if queue.get('paused') or queue.get('status')=='stopped':raise RuntimeError('Campaign stopped/paused; no audit of incomplete evidence')
            if not wait:raise RuntimeError('Campaign incomplete')
            state['heartbeat_unix']=time.time();state['active_phase']=queue.get('active_phase');save(audit/'status.json',state);time.sleep(30)
        plan=read(root/'plan.json');extension=read(root/'refinement_extension.json');parents=read(root/'parents.json')
        binary=REPO/'code/zk/bin/commitjl-bound-referee-final-20261025.exe';extra_binary=Path(extension['binary'])
        assert digest(binary)==plan['binary_sha256'] and digest(extra_binary)==extension['binary_sha256']
        def public_copy(source,target):
            manifest=read(source/'public_manifest.json')['sha256']
            assert all(digest(source/name)==h for name,h in manifest.items())
            assert not any(p.suffix in ['.pk','.gob'] or p.name=='input.json' for p in source.rglob('*') if p.is_file())
            if not target.exists():shutil.copytree(source,target)
            assert all(digest(target/name)==h for name,h in manifest.items())
            return manifest
        def replay(label,source,pin,task,phase,expected,batch=16,extra=None):
            assert phase in ['verify','group_verify','moments_verify','refinement_verify'],'Verification-only audit'
            target=audit/label;manifest=public_copy(source,target);exe=extra_binary if extra else binary
            command=[str(exe),'-phase',phase,'-out',str(target),'-q',str(batch),'-trust',str(pin),'-public-task',str(task)]
            if extra:
                base,refpin=extra;basecopy=audit/(label+'_base');public_copy(base,basecopy)
                command+=['-refinement-base',str(basecopy),'-refinement-trust',str(refpin)]
            state.update(status='replaying',active_phase=label);save(audit/'status.json',state);started=time.monotonic()
            with (audit/(label+'.log')).open('wb') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=env)
            receipt=read(target/(phase+'.json'));assert result.returncode==0 and receipt['complete'] and receipt['verified']==expected,(label,result.returncode,receipt)
            assert all(digest(target/name)==h for name,h in manifest.items())
            state['results'].append(dict(label=label,verified=expected,binary_sha256=digest(exe),source=str(source),source_manifest_sha256=digest(source/'public_manifest.json'),replay_receipt_sha256=digest(target/(phase+'.json')),wall_seconds=time.monotonic()-started));save(audit/'status.json',state)
        for case in plan['cases']:
            tag=case['id'];parent=Path(parents[tag]['ordered_head']);p=parent/'full'
            verify=read(p/'public_bundle/verify.json')
            replay(tag+'_full',p/'public_bundle',p/'owner_pin.json',p/'approved_public_task.json','verify',verify['verified'],8)
            for variant in ['head79','pca79']:
                base=root/tag/variant;ex=Path(extension['root'])/tag/variant;r=read(ex/'checked_comparison.json')
                assert r['complete']
                if r['new_refinement_proofs']:
                    replay(tag+'_'+variant,ex/'public',base/'owner_pin.json',base/'approved.json','refinement_verify',r['base_public_proofs']+r['new_refinement_proofs'],extra=(base/'public',ex/'owner_refinement_pin.json'))
                else:replay(tag+'_'+variant,base/'public',base/'owner_pin.json',base/'approved.json','group_verify',r['base_public_proofs'])
                if r['fresh_fallback_proofs']:
                    fb=ex/'fallback/full';replay(tag+'_'+variant+'_fallback',fb/'public_bundle',fb/'owner_pin.json',fb/'approved_public_task.json','verify',r['fresh_fallback_proofs'],8)
            moments=root/tag/'moments_only';r=read(moments/'checked_comparison.json')
            replay(tag+'_moments',moments/'public',moments/'owner_pin.json',moments/'approved.json','moments_verify',r['group_public_proofs'])
            if r['fresh_fallback_proofs']:
                fb=moments/'fallback/full';replay(tag+'_moments_fallback',fb/'public_bundle',fb/'owner_pin.json',fb/'approved_public_task.json','verify',r['fresh_fallback_proofs'],8)
            affine=root/tag/'known_family_affine/family_affine';r=read(root/tag/'known_family_affine/checked_comparison.json')
            replay(tag+'_affine',affine/'public_bundle',affine/'owner_pin.json',affine/'approved_public_task.json','verify',r['family']['public_proofs'],16)
        repeat=root/'timing_repeat'
        if (repeat/'checked_timing_repeat.json').exists():
            result=read(repeat/'checked_timing_repeat.json');assert result['complete'] and result['proof_hash_sets_disjoint']
            base=repeat/'head79';ex=repeat/'refined';r=read(ex/'checked_comparison.json')
            replay('mnist_mature_timing_repeat_head',ex/'public',base/'owner_pin.json',base/'approved.json','refinement_verify',r['base_public_proofs']+r['new_refinement_proofs'],extra=(base/'public',ex/'owner_refinement_pin.json'))
        state.update(complete=True,status='complete',verified_total=sum(r['verified'] for r in state['results']),finished_unix=time.time(),scope=__doc__);save(audit/'status.json',state)
    except BaseException as error:
        state.update(status='stopped',error=str(error));save(audit/'status.json',state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--wait',action='store_true');p.add_argument('--audit-root',type=Path);p.add_argument('--workers',type=int,default=1);p.add_argument('--memory-mib',type=int,default=256);a=p.parse_args();run(a.root.resolve(),a.wait,a.audit_root.resolve() if a.audit_root else None,a.workers,a.memory_mib)
