"""Replay public proofs, then apply an explicit per-center-set accuracy policy.

Exit 0: cryptographic replay and requested policy pass. Exit 3: proofs are valid
but certificate quality is insufficient. Original proof/result files are read
only; replay receipts are written into a fresh audit directory.
"""
from pathlib import Path
import argparse,hashlib,json,re,subprocess,sys

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from experiments.review_controls import public_export


def audit(a):
    a.out.mkdir(parents=True,exist_ok=False)
    stage=a.out/'public_bundle';public_export(a.bundle,stage)
    command=[str(a.bin.resolve()),'-phase','verify','-out',str(stage.resolve()),
             '-trust',str(a.trust.resolve()),'-public-task',str(a.task.resolve())]
    process=subprocess.run(command,capture_output=True,text=True,timeout=a.timeout)
    (a.out/'replay_stdout.txt').write_text(process.stdout,encoding='utf-8')
    (a.out/'replay_stderr.txt').write_text(process.stderr,encoding='utf-8')
    result=dict(cryptographic_replay_valid=False,accuracy_policy_accepted=False,
        policy=dict(min_certified_bps=a.min_certified_bps,max_cost_width_bps=a.max_cost_width_bps,
                    require_whole_catalog=not a.allow_subset),per_set=[])
    if process.returncode!=0:
        result['error']='cryptographic replay failed';return result,2
    receipt=json.loads((stage/'verify.json').read_text())
    if not receipt.get('complete'):raise RuntimeError('incomplete replay receipt')
    result['cryptographic_replay_valid']=True
    trust=json.loads(a.trust.read_text());n=trust['model']['n'];t=int(trust['model']['basis_scale'])
    task=json.loads(a.task.read_text());found=0
    for key in sorted(trust['verifying_key_sha256']):
        match=re.fullmatch(r'reuse_certified_(\d+)_k(\d+)',key)
        if not match:continue
        found+=1;q=int(match[1]);phase=json.loads((stage/f'reuse_certified_{q}.json').read_text())
        for s in range(len(task['reuse_centers'])):
            samples=[x for x in phase['samples'] if x.get('centre_set',0)==s]
            indices=[i for x in samples for i in x['indices']]
            if len(indices)!=len(set(indices)):raise RuntimeError('policy aggregation repeats records')
            if not a.allow_subset and sorted(indices)!=list(range(n)):raise RuntimeError('whole-catalog policy lacks coverage')
            count=sum(x.get('certified_count',0) for x in samples)
            lo=sum(int(x['cost_lower_scaled']) for x in samples);hi=sum(int(x['cost_upper_scaled']) for x in samples)
            point=sum(int(x['psi']) for x in samples)*t*t
            if not lo<=point<=hi:raise RuntimeError('quoted estimate not inside verified interval')
            count_ok=10000*count>=a.min_certified_bps*len(indices)
            cost_ok=10000*(hi-lo)<=a.max_cost_width_bps*lo
            result['per_set'].append(dict(key=key,center_set=s,requested_rows=len(indices),certified_rows=count,
                certified_fraction=count/len(indices),cost_lower_scaled=str(lo),cost_upper_scaled=str(hi),
                quoted_cost_scaled=str(point),cost_denominator=str(t**6),input_scale=trust['model']['input_scale'],
                interval_relative_width=(hi-lo)/lo if lo else (0 if hi==0 else None),
                certified_fraction_pass=count_ok,cost_precision_pass=cost_ok,accepted=count_ok and cost_ok))
    if not found:raise RuntimeError('no accuracy certificates; execution-only proofs do not satisfy this policy')
    result['accuracy_policy_accepted']=all(x['accepted'] for x in result['per_set'])
    result['trusted_inputs_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.trust,a.task]}
    return result,0 if result['accuracy_policy_accepted'] else 3


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bin',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--trust',type=Path,required=True);p.add_argument('--task',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--min-certified-bps',type=int,default=9900)
    p.add_argument('--max-cost-width-bps',type=int,default=100);p.add_argument('--allow-subset',action='store_true')
    p.add_argument('--timeout',type=int,default=3600);a=p.parse_args()
    if not 0<=a.min_certified_bps<=10000 or a.max_cost_width_bps<0:p.error('invalid acceptance policy')
    result,code=audit(a)
    (a.out/'acceptance.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(cryptographic_replay_valid=result['cryptographic_replay_valid'],accuracy_policy_accepted=result['accuracy_policy_accepted'],exit_code=code)),flush=True)
    raise SystemExit(code)


if __name__=='__main__':main()
