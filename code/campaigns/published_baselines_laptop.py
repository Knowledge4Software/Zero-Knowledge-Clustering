"""Serial matched Linux/WSL comparison of a published-ZKML primitive adapter.

No paper export. Same frozen original integers, all candidate banks, eight
workers and CPU affinity for both implementations. Full and compact paths
are generated afresh in Linux, not compared with historical Windows times.
"""
from pathlib import Path
import argparse,copy,hashlib,json,os,shutil,subprocess,sys,time

REPO=Path(__file__).resolve().parents[2]
CODE=REPO/'code'
DEFAULT=CODE/'validation/outputs/published_baselines_20261027'
RUST=Path(os.environ.get('COMMITJL_ZKML_BINARY', '/home/koush/.cache/commitjl-external/zkml-target/release/prototype_audit'))
GO=Path(os.environ.get('COMMITJL_GO_BINARY', str(REPO/'tmp/external_baselines_20261027/commitjl-bound-linux')))


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2),encoding='utf-8');t.replace(p)


class Campaign:
    def __init__(self,root):
        self.root=root;self.state=read(root/'matched_queue.json') if (root/'matched_queue.json').exists() else dict(started_unix=time.time(),complete=False,completed=[])
        self.env=dict(os.environ,PYTHONPATH=str(CODE),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',RAYON_NUM_THREADS='8',GOMAXPROCS='8',GOMEMLIMIT='1200MiB')
    def invoke(self,label,command,timeout=1800,expect_failure=False):
        folder=self.root/'resources';folder.mkdir(exist_ok=True)
        previous=sorted(folder.glob(label+'.attempt_*.json'))
        if previous and read(previous[-1]).get('complete'):
            return read(previous[-1])
        if (self.root/'STOP').exists():raise RuntimeError('User stop requested')
        if time.time()-self.state['started_unix']>6*3600:raise RuntimeError('Six-hour bounded campaign limit')
        attempt=folder/f'{label}.attempt_{len(previous)+1:02}.json';log=attempt.with_suffix('.log')
        self.state.update(active_phase=label,status='running');save(self.root/'matched_queue.json',self.state)
        args=[sys.executable,str(CODE/'tools/guarded_linux_run.py'),'--receipt',str(attempt),'--log',str(log),'--timeout',str(timeout)]
        if expect_failure:args+=['--expect-failure']
        args+=['--']+list(map(str,command))
        r=subprocess.run(args,cwd=REPO,env=self.env)
        if r.returncode:raise RuntimeError(f'{label} failed: inspect {log}')
        value=read(attempt)
        if label not in self.state['completed']:self.state['completed'].append(label)
        save(self.root/'matched_queue.json',self.state);return value

    def smoke(self):
        root=self.root/'smoke_v2';root.mkdir(exist_ok=True)
        source=read(self.root/'smoke/input.json');save(root/'input.json',source)
        save(root/'approved_task.json',dict(reuse_centers=source['reuse_centers']))
        bundle=root/'private';public=root/'public';public.mkdir(exist_ok=True)
        self.invoke('zkml_smoke_prepare',[RUST,'prepare',root/'input.json',bundle,12,17],300)
        self.invoke('zkml_smoke_selftest',[RUST,'selftest',root/'input.json',bundle],300)
        self.invoke('zkml_smoke_query',[RUST,'query',root/'input.json',bundle,1],300)
        for name in ['params','vk']:
            shutil.copyfile(bundle/name,public/name)
        shutil.copytree(bundle/'bank_001',public/'bank_001',dirs_exist_ok=True)
        for name in ['private_diagnostics.json','result.json']:(public/'bank_001'/name).unlink(missing_ok=True)
        shutil.copyfile(bundle/'owner_pin.json',root/'owner_pin.json')
        args=[RUST,'verify',public,1,root/'owner_pin.json',root/'approved_task.json']
        self.invoke('zkml_smoke_public_verify',args,300)
        # Keep honest public material intact while checking four substitutions.
        proof=public/'bank_001/proof';original=proof.read_bytes()
        try:
            proof.write_bytes(bytes([original[0]^1])+original[1:]);self.invoke('zkml_smoke_bad_proof',args,300,True)
        finally:proof.write_bytes(original)
        statement=public/'bank_001/public.json';original=read(statement)
        try:
            altered=copy.deepcopy(original);altered['cost']+=1
            altered['public'][-1]=altered['cost'].to_bytes(32,'little').hex()
            save(statement,altered);self.invoke('zkml_smoke_bad_cost',args,300,True)
        finally:save(statement,original)
        task=root/'approved_task.json';original=read(task)
        try:
            altered=copy.deepcopy(original);altered['reuse_centers'][1][0][0]+=1
            save(task,altered);self.invoke('zkml_smoke_bad_centers',args,300,True)
        finally:save(task,original)
        pin=root/'owner_pin.json';original=read(pin)
        try:
            altered=copy.deepcopy(original);altered['catalog_commitment']='01'+altered['catalog_commitment'][2:]
            save(pin,altered);self.invoke('zkml_smoke_bad_root',args,300,True)
        finally:save(pin,original)
        self.invoke('zkml_smoke_restored',args,300)
        assert not list(public.rglob('*.pk')) and not (public/'pk').exists() and not (public/'input.json').exists()
        save(root/'checks.json',dict(complete=True,real_proof=True,public_only=True,
            real_verifier_tamper_checks=4,negative_witness_checks=4,rust_binary_sha256=sha(RUST)))
        self.invoke('compact_smoke',[sys.executable,'-m','validation.check_group_fast_proofs','--root',self.root/'compact_smoke','--binary',GO,'--compact'],600)
        self.invoke('refinement_smoke',[sys.executable,'-m','validation.check_group_refinement_proofs','--root',self.root/'refinement_smoke','--binary',GO],600)

    def prepare_case(self,name,source,n):
        import numpy as np
        from commitjl.pilot_io import array_sha
        target=self.root/'matched'/f'{name}_n{n}'
        for basis in ['ordered_head','pca']:
            inp=read(source/basis/'head_r63/input.json');inp['model']['n']=n;inp['x']=inp['x'][:n]
            inp['queries']=[list(range(i,min(n,i+16))) for i in range(0,n,16)]
            save(target/basis/'head_r63/input.json',inp)
            full=copy.deepcopy(inp);full['model'].update(method='full',basis=[],seed='')
            full['queries']=[list(range(i,min(n,i+8))) for i in range(0,n,8)]
            save(target/basis/'full/input.json',full)
            protocol=read(source/basis/'protocol.json');protocol.update(n=n,proof_runtime_workers=8,
                phase_wall_limit_seconds=1800,execution_os='Linux/WSL',fitting_seconds=0.,
                source_input_sha256=sha(source/basis/'head_r63/input.json'),
                proof_audit_sha256=array_sha(np.asarray(inp['x'],dtype=np.int64)),
                source_selection='First fixed prefix, every supplied distinct candidate; no outcome-based selection.')
            save(target/basis/'protocol.json',protocol)
            aux=read(source/basis/'auxiliary_basis.json');aux['source_input_sha256']=sha(target/basis/'head_r63/input.json');aux['fitting_seconds']=0.
            save(target/basis/'auxiliary_basis.json',aux)
        left=read(target/'ordered_head/full/input.json');right=read(target/'pca/full/input.json')
        assert left['x']==right['x'] and left['reuse_centers']==right['reuse_centers']
        save(target/'approved_task.json',dict(reuse_centers=left['reuse_centers']))
        save(target/'input_pin.json',dict(complete=True,n=n,d=left['model']['d'],k=left['model']['k'],
            source=str(source),full_input_sha256=sha(target/'ordered_head/full/input.json'),
            centroid_input_sha256=sha(target/'ordered_head/head_r63/input.json'),
            pca_input_sha256=sha(target/'pca/head_r63/input.json'),centers=len(left['reuse_centers'])))
        return target

    def rust_case(self,target):
        name=target.name;source=target/'ordered_head/full/input.json';bundle=target/'zkml_private';public=target/'zkml_public';public.mkdir(exist_ok=True)
        inspection=target/'zkml_inspection'
        # Circuit degree depends only on whether all rows fit, not prediction quality.
        if not (inspection/'selection.json').exists():
            trials=[]
            for k in range(12,19):
                prior=sorted((self.root/'resources').glob(name+f'_zkml_inspect_k{k}.attempt_*.log'))
                if prior and any(t in prior[-1].read_text(errors='replace') for t in ['NotEnoughRowsAvailable','usable_rows=']):
                    trials.append(dict(k=k,fit=False));continue
                try:
                    self.invoke(name+f'_zkml_inspect_k{k}',[RUST,'inspect',source,inspection/f'k{k}',k,17],300)
                    trials.append(dict(k=k,fit=True));save(inspection/'selection.json',dict(k=k,trials=trials,selection='Smallest declared degree satisfying all constraints at fixed 17 advice columns.'));break
                except RuntimeError:
                    attempt=sorted((self.root/'resources').glob(name+f'_zkml_inspect_k{k}.attempt_*.log'))[-1]
                    if not any(t in attempt.read_text(errors='replace') for t in ['NotEnoughRowsAvailable','usable_rows=']):
                        raise
                    trials.append(dict(k=k,fit=False))
            else:raise RuntimeError('No feasible declared ZKML circuit degree')
        k=read(inspection/'selection.json')['k']
        self.invoke(name+'_zkml_prepare',[RUST,'prepare',source,bundle,k,17],1800)
        shutil.copyfile(bundle/'owner_pin.json',target/'zkml_owner_pin.json')
        for file in ['params','vk']:shutil.copyfile(bundle/file,public/file)
        banks=read(source)['reuse_centers'];costs=[];proof_bytes=0
        self.invoke(name+'_zkml_queries',[RUST,'queries',source,bundle],1800)
        for i,_ in enumerate(banks):
            private_bank=bundle/f'bank_{i:03}';public_bank=public/f'bank_{i:03}';public_bank.mkdir(exist_ok=True)
            for file in ['proof','public.json']:shutil.copyfile(private_bank/file,public_bank/file)
            self.invoke(name+f'_zkml_verify{i:03}',[RUST,'verify',public,i,target/'zkml_owner_pin.json',target/'approved_task.json'],300)
            result=read(private_bank/'result.json');costs.append(result['cost']);proof_bytes+=result['proof_bytes']
        # Independently recompute every full-catalog exact cost with Python integers.
        inp=read(source);labels=[]
        for i,centers in enumerate(banks):
            exact=0;labs=[]
            for row in inp['x']:
                ds=[sum((int(x)-int(c))**2 for x,c in zip(row,center)) for center in centers]
                j=min(range(len(ds)),key=ds.__getitem__);labs.append(j);exact+=ds[j]
            assert exact==costs[i]
            assert labs==read(bundle/f'bank_{i:03}/private_diagnostics.json')['labels']
            labels.append(labs)
        save(target/'zkml_checked.json',dict(complete=True,costs=costs,changes=[c-costs[0] for c in costs[1:]],
            all_costs_match_integer_oracle=True,assignment_agreement_percent=100.,
            proofs=len(banks),proof_bytes=proof_bytes,public_only_replay=True,
            public_manifest={p.relative_to(public).as_posix():sha(p) for p in public.rglob('*') if p.is_file()},
            adaptation='Published ZKML exact dot-product, adder and Poseidon primitives with explicit integer audit wrapper; not the unmodified optimizing compiler.'))

    def method_case(self,target):
        name=target.name;parent=target/'ordered_head';pca=target/'pca'
        self.invoke(name+'_groth_full',[sys.executable,'-m','campaigns.local_fast_tradeoff','--root',parent,'--binary',GO,'--run','--cases','full'],1800)
        from campaigns.shared_full_reference import share_full_reference
        share_full_reference(parent,pca)
        for method,p in [('ordered_head',parent),('pca',pca)]:
            base=target/(method+'_base');refined=target/(method+'_refined')
            self.invoke(name+'_'+method+'_base',[sys.executable,'-m','campaigns.run_group_fast_control','--compact','--proof-backbone-only',
                '--parent',p,'--root',base,'--binary',GO,'--smoke-receipt',self.root/'compact_smoke/checks.json',
                '--auxiliary-basis',p/'auxiliary_basis.json'],1800)
            self.invoke(name+'_'+method+'_refined',[sys.executable,'-m','campaigns.run_group_refinement_control','--base',base,
                '--parent',p,'--root',refined,'--binary',GO,'--smoke-receipt',self.root/'refinement_smoke/checks.json'],1800)

    def run(self):
        assert os.name=='posix' and RUST.exists() and GO.exists()
        try:
            self.smoke()
            cases=[('fashion',CODE/'validation/outputs/referee_optimized_20261026/extended/fashion_mnist_k5_seed173_age8_n1024/repetition_01'),
                ('text',CODE/'validation/outputs/referee_two_reviews_20261027/learning_inputs/text_minilm384_k5_seed307_release')]
            for name,source in cases:
                target=self.prepare_case(name,source,32)
                self.rust_case(target)
                self.method_case(target)
            self.state.update(complete=True,status='complete',finished_unix=time.time(),active_phase=None)
            save(self.root/'matched_queue.json',self.state)
        except BaseException as error:
            self.state.update(complete=False,status='stopped_with_preserved_evidence',error=str(error));save(self.root/'matched_queue.json',self.state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT);a=p.parse_args()
    Campaign(a.root.resolve()).run()
