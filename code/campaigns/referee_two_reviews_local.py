"""Serial, resource-limited experiments authorized for the two new reviews."""
from pathlib import Path
import argparse,copy,json,os,subprocess,sys,time
import psutil
from campaigns.local_fast_tradeoff import save,sha
from campaigns.execution_admission import require_execution
from campaigns.proof_runtime import memory_stop

REPO=Path(__file__).resolve().parents[2]
POLICY=REPO/'code/validation/outputs/referee_round2_20261026/execution_policy.json'
OLD=REPO/'code/validation/outputs/referee_optimized_20261026'


def read(path):return json.loads(Path(path).read_text())


class Runner:
    def __init__(self,root):
        self.root=root;self.plan=read(root/'plan.json');self.state=read(root/'queue.json')
        if self.state.get('supervisor_pid') and psutil.pid_exists(self.state['supervisor_pid']):raise RuntimeError('Existing review supervisor is still live')
        if self.state['plan_sha256']!=sha(root/'plan.json'):raise RuntimeError('Review plan changed after fitting')
        if self.plan['maximum_proof_catalog_records']>1024:raise RuntimeError('Laptop proof scope exceeds user limit')
        if self.state.get('error'):
            self.state.setdefault('restarts',[]).append(dict(recorded_unix=time.time(),previous_error=self.state.pop('error')))
        self.state.update(supervisor_pid=os.getpid(),status='running');save(root/'queue.json',self.state)
        self.state.setdefault('budget_started_unix',self.state['started_unix']);save(root/'queue.json',self.state)
        self.deadline=self.state['budget_started_unix']+self.plan['total_proof_budget_hours']*3600
        self.py=[sys.executable,'-u','-m'];self.binary=REPO/'code/zk/bin/commitjl-bound-final-optimized-20261026.exe'
        self.plonk=REPO/self.plan['plonk_binary'];assert sha(self.plonk)==self.plan['plonk_binary_sha256']
        self.env=dict(os.environ,PYTHONPATH=str(REPO/'code'),COMMITJL_EXECUTION_TARGET='laptop',COMMITJL_EXECUTION_POLICY=str(POLICY),
            COMMITJL_MEMORY_PRESSURE_POLICY='windows_os',GOMAXPROCS='8',GOMEMLIMIT='1200MiB',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        self.state.setdefault('attempts',[])

    def monitor(self,process,started,limit):
        peak=0;stop=None;birth=process.create_time()
        while process.is_running():
            try:
                if process.create_time()!=birth:raise RuntimeError('PID reused')
                tree=[process]+process.children(recursive=True)
                rss=max((p.memory_info().rss for p in tree),default=0);peak=max(peak,rss);stop=memory_stop(rss)
                try:require_execution(read(POLICY),target='laptop')
                except RuntimeError:stop='execution authorization held'
                if (self.root/'STOP').exists() or (self.root.parent/'STOP').exists():stop='user stop file'
                if psutil.disk_usage(str(self.root)).free<2**30:stop='disk reserve'
                if time.time()>self.deadline:stop='24hour campaign budget'
                if time.time()-started>limit:stop='phase wall limit'
                if stop:
                    for p in reversed(tree):
                        try:p.kill()
                        except psutil.NoSuchProcess:pass
                    break
                time.sleep(.5)
            except psutil.NoSuchProcess:break
        return peak,stop

    def invoke(self,label,args,limit=12*3600,resource_exclusion=False):
        if label in self.state['completed']:return True
        require_execution(read(POLICY),target='laptop')
        if (self.root/'STOP').exists():raise RuntimeError('User stop file present')
        started=time.time();self.state.update(active_phase=label,status='running',child_pid=None);save(self.root/'queue.json',self.state)
        retries=sum(v['phase']==label for v in self.state['attempts'])
        log_path=self.root/(label+('' if not retries else f'.retry{retries:02d}')+'.log')
        with log_path.open('xb') as log:
            child=subprocess.Popen(args,cwd=REPO,env=self.env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.state['child_pid']=child.pid;save(self.root/'queue.json',self.state)
            peak,stop=self.monitor(psutil.Process(child.pid),started,limit);rc=child.wait()
        attempt=dict(phase=label,wall_seconds=time.time()-started,returncode=rc,stop=stop,peak_rss_bytes=peak,log=log_path.name)
        self.state['attempts'].append(attempt);self.state['child_pid']=None
        if not rc and not stop:self.state['completed'].append(label)
        save(self.root/'queue.json',self.state)
        if rc or stop:
            if resource_exclusion and stop in ['RAM reserve','process RSS cap','disk reserve','phase wall limit']:
                save(self.root/(label+'_resource_exclusion.json'),attempt);return False
            raise RuntimeError(label+' failed; inspect its preserved log: '+str(stop))
        return True

    def wait_native(self):
        if (self.root/'learning_native/summary.json').exists():
            if not read(self.root/'learning_native/summary.json')['complete']:raise RuntimeError('Incomplete native summary')
        else:
            identity=self.state['active_processes'][0];p=psutil.Process(identity['pid'])
            if abs(p.create_time()-identity['created_unix'])>.01 or p.cmdline()!=identity['command']:raise RuntimeError('Native worker identity changed')
            self.state.update(active_phase='learning_native',child_pid=p.pid);save(self.root/'queue.json',self.state)
            peak,stop=self.monitor(p,p.create_time(),3600)
            if stop or not (self.root/'learning_native/summary.json').exists():raise RuntimeError('Native workflow did not finish: '+str(stop))
            assert read(self.root/'learning_native/summary.json')['complete']
        if 'learning_native' not in self.state['completed']:self.state['completed'].append('learning_native')
        save(self.root/'queue.json',self.state)

    def subset(self,source,target,n,sets=None):
        target.mkdir(parents=True,exist_ok=True)
        def unchanged(path,value):
            if path.exists():
                if read(path)!=value:raise RuntimeError('Existing subset input differs: '+str(path))
            else:save(path,value)
        for method in ['ordered_head','pca']:
            for name in ['head_r63/input.json','full/input.json']:
                value=read(source/method/name);value['model']['n']=n;value['x']=value['x'][:n]
                if sets is not None:value['reuse_centers']=value['reuse_centers'][:sets]
                q=16 if name.startswith('head') else 8
                value['queries']=[list(range(i,min(n,i+q))) for i in range(0,n,q)]
                unchanged(target/method/name,value)
            protocol=read(source/method/'protocol.json');protocol.update(n=n,phase_wall_limit_seconds=12*3600)
            if sets is not None:protocol['selected_source_sets']=protocol['selected_source_sets'][:sets-1]
            unchanged(target/method/'protocol.json',protocol)
            aux=read(source/method/'auxiliary_basis.json');aux['source_input_sha256']=sha(target/method/'head_r63/input.json');unchanged(target/method/'auxiliary_basis.json',aux)
        return target

    def learning_proofs(self,source,target):
        from campaigns.shared_full_reference import share_full_reference
        parent=source/'ordered_head';pca=source/'pca'
        if not self.invoke(target.name+'_full',self.py+['campaigns.local_fast_tradeoff','--root',str(parent),'--binary',str(self.binary),'--run','--cases','full'],resource_exclusion=True):return False
        share_full_reference(parent,pca)
        for method,p in [('ordered_head',parent),('pca',pca)]:
            base=target/(method+'_base');refined=target/(method+'_refined')
            if not self.invoke(target.name+'_'+method+'_base',self.py+['campaigns.run_group_fast_control','--compact','--proof-backbone-only','--parent',str(p),
                '--root',str(base),'--binary',str(self.binary),'--smoke-receipt',str(OLD/'compact_smoke/checks.json'),'--auxiliary-basis',str(p/'auxiliary_basis.json')],resource_exclusion=True):return False
            if not self.invoke(target.name+'_'+method+'_refined',self.py+['campaigns.run_group_refinement_control','--base',str(base),'--parent',str(p),'--root',str(refined),
                '--binary',str(self.binary),'--smoke-receipt',str(OLD/'refinement_smoke/checks.json')],resource_exclusion=True):return False
        return True

    def run(self):
        try:
            self.wait_native()
            source=self.root/'learning_inputs/text_minilm384_k5_seed307_release'
            # This small first-prefix pilot selects only a feasible backend
            # workload size by runtime, never by accuracy/certificate outcome.
            pilot=self.subset(source,self.root/'backend_pilot_inputs',8,sets=2)
            inp=pilot/'ordered_head/full/input.json';out=self.root/'backend_pilot';cache=self.root/'unsafe_test_srs_cache'
            args=[str(self.plonk),'-phase','run','-input',str(inp),'-out',str(out),'-q','1','-allow-benchmark-srs','-srs-cache',str(cache)]
            if self.invoke('backend_pilot',args,limit=1800,resource_exclusion=True):
                self.invoke('backend_pilot_verify',[str(self.plonk),'-phase','verify','-out',str(out),'-trust',str(out/'owner_pin.json'),'-public-task',str(out/'approved_task.json')],limit=600)
                result=read(out/'result.json');per=sum(v['prove_seconds'] for v in result['samples'])/len(result['samples'])
                sets=len(read(source/'ordered_head/full/input.json')['reuse_centers'])
                n=next((v for v in [128,64,32] if v*sets*per<3600),32)
                chosen=dict(n=n,estimated_proving_seconds=n*sets*per,pilot_mean_seconds_per_proof=per,
                    selection='Largest declared <=128 prefix with projected backend proving <=1hour; chosen only from pilot runtime. Same catalog/requests for compared methods; all quality outcomes retained.')
                save(self.root/'backend_workload_selection.json',chosen)
                selected=self.subset(source,self.root/'backend_matched_inputs',n)
                full_input=selected/'ordered_head/full/input.json'
                self.invoke('backend_inspect',[str(self.plonk),'-phase','inspect','-input',str(full_input),'-out',str(self.root/'backend_inspect'),'-q','1'],limit=300,resource_exclusion=True)
                self.invoke('backend_srs_preparation',[str(self.plonk),'-phase','run','-input',str(full_input),'-out',str(self.root/'backend_srs_preparation'),'-q','1','-allow-benchmark-srs','-srs-cache',str(cache),'-prepare-srs-only'],limit=1800,resource_exclusion=True)
                for rep in range(1,4):
                    out=self.root/'backend'/f'repetition_{rep:02d}'
                    out.parent.mkdir(parents=True,exist_ok=True)
                    if not self.invoke(f'backend_r{rep}',[str(self.plonk),'-phase','run','-input',str(full_input),'-out',str(out),'-q','1','-allow-benchmark-srs','-srs-cache',str(cache)],limit=7200,resource_exclusion=True):break
                    self.invoke(f'backend_r{rep}_verify',[str(self.plonk),'-phase','verify','-out',str(out),'-trust',str(out/'owner_pin.json'),'-public-task',str(out/'approved_task.json')],limit=600)
                self.learning_proofs(selected,self.root/'backend_groth_controls')
            for case in self.plan['proof_cases']:
                fit=f"{case['dataset']}_k{case['k']}_seed{case['seed']}_release"
                self.learning_proofs(self.root/'learning_inputs'/fit,self.root/('proof_'+case['dataset']))
            self.state.update(complete=True,status='review_proofs_complete',child_pid=None,finished_unix=time.time());save(self.root/'queue.json',self.state)
        except BaseException as e:
            self.state.update(status='stopped',error=str(e),child_pid=None);save(self.root/'queue.json',self.state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();Runner(a.root.resolve()).run()
