"""Public-only update decisions with whole-task exact fallback when needed.

Requires separately approved owner/setup pins and tasks. No original rows,
assignments, compressed private statistics, or per-row certificate flags are
read. A fallback reveals exact aggregate costs for its requested center sets;
it never reveals individual uncertain-record identities.
"""
from pathlib import Path
import argparse,copy,json,shutil,subprocess,time
from experiments.bound_audit import save


def audit(a):
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    pair_pin=json.loads(a.paired_pin.read_text());full_pin=json.loads(a.full_pin.read_text())
    task=json.loads(a.paired_task.read_text());full_task=json.loads(a.full_task.read_text())
    if pair_pin['original_root']!=full_pin['original_root']:raise ValueError('different committed catalogs')
    for key in ['n','d','input_bits','input_scale']:
        if pair_pin['model'][key]!=full_pin['model'][key]:raise ValueError('quantized task mismatch')
    if full_pin['model']['method']!='full':raise ValueError('fallback must use exact Full')
    if task['reuse_centers']!=full_task['reuse_centers'] or task['queries']!=full_task['queries']:
        raise ValueError('fallback source task differs from approved paired requests')
    if task['reuse_centers'][0]!=task['comparison_base']:raise ValueError('baseline cost proof must be available as first approved set')
    def verify(bundle,pin,approved,label):
        cmd=[str(a.bin.resolve()),'-phase','verify','-out',str(bundle.resolve()),'-trust',str(pin.resolve()),'-public-task',str(approved.resolve())]
        with (a.out/(label+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(label+' public proof verification failed')
        data=json.loads((bundle/'verify.json').read_text())
        if not data['complete']:raise ValueError('incomplete public replay')
        save(a.out/(label+'.json'),data)
    # Copy only the already public export: preserve original replay receipts.
    public_pair=a.out/'paired_public';shutil.copytree(a.paired_bundle,public_pair)
    verify(public_pair,a.paired_pin,a.paired_task,'paired_verify')
    q=len(task['queries'][0])
    if any(len(indices)!=q for indices in task['queries']):raise ValueError('composed auditor requires uniform approved batch size')
    phase=json.loads((public_pair/f'paired_change_{q}.json').read_text())
    verdicts=[];fallback=[]
    for i in range(len(task['reuse_centers'])):
        samples=[s for s in phase['samples'] if s.get('centre_set',0)==i]
        lo=sum(int(s['change_lower_scaled']) for s in samples);hi=sum(int(s['change_upper_scaled']) for s in samples)
        verdict='improves' if hi<0 else ('worsens' if lo>0 else ('equal' if lo==hi==0 else 'inconclusive'))
        verdicts.append(dict(set=i,paired_decision=verdict,final_decision=verdict,used_exact_fallback=False,
                             change_lower_scaled=str(lo),change_upper_scaled=str(hi)))
        if verdict=='inconclusive':fallback.append(i)
    exact_sets=[]
    if fallback:
        exact_sets=[0]+[i for i in fallback if i!=0]
        public_full=a.out/'selected_full_public';public_full.mkdir()
        shutil.copy2(a.full_bundle/'prepare.json',public_full/'prepare.json')
        shutil.copytree(a.full_bundle/'keys',public_full/'keys')
        data=json.loads((a.full_bundle/f'full_reuse_{q}.json').read_text());selected=[]
        # Select actual saved proofs; metadata ordinal/file changes do not change
        # their public mathematical statement. The verifier rebuilds that statement
        # from the independently approved selected center coordinates and indices.
        for ordinal,original_set in enumerate(exact_sets):
            samples=[s for s in data['samples'] if s.get('centre_set',0)==original_set]
            if len(samples)!=len(task['queries']):raise ValueError('missing exact fallback batches')
            samples=sorted(samples,key=lambda s:s['id'])
            for query,s in enumerate(samples):
                oldfile=f'full_reuse/s{original_set:03d}_q{q}_{query:04d}.proof'
                if s['file']!=oldfile or s['indices']!=task['queries'][query]:raise ValueError('unapproved fallback source proof identity')
                newfile=f'full_reuse/s{ordinal:03d}_q{q}_{query:04d}.proof'
                dest=public_full/newfile;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(a.full_bundle/oldfile,dest)
                sample=copy.deepcopy(s);sample.update(centre_set=ordinal,id=query,file=newfile);selected.append(sample)
        data['samples']=selected;data['verified']=len(selected)
        save(public_full/f'full_reuse_{q}.json',data)
        selected_task={**task,'reuse_centers':[task['reuse_centers'][i] for i in exact_sets]}
        selected_path=a.out/'approved_fallback_task.json';save(selected_path,selected_task)
        verify(public_full,a.full_pin,selected_path,'fallback_verify')
        costs={original:sum(int(s['psi']) for s in selected if s.get('centre_set',0)==ordinal) for ordinal,original in enumerate(exact_sets)}
        for i in fallback:
            delta=costs[i]-costs[0]
            verdicts[i].update(final_decision='improves' if delta<0 else ('worsens' if delta>0 else 'equal'),
                               used_exact_fallback=True,exact_public_cost_change=str(delta))
    save(a.out/'decisions.json',dict(complete=True,decisions=verdicts,exact_center_sets_proved=exact_sets,
        public_only=True,original_root=pair_pin['original_root'],auditor_workflow_wall_seconds=time.monotonic()-started,
        privacy='Paired interval outputs plus exact aggregate baseline/candidate costs for fallback tasks; no per-row assignments or uncertain-row indices.',
        trust_boundary='Owner/setup pins and approved task must be obtained independently of the prover.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['paired-pin','paired-task','paired-bundle','full-pin','full-task','full-bundle','out','bin']:
        p.add_argument('--'+name,type=Path,required=True)
    audit(p.parse_args())


if __name__=='__main__':main()
