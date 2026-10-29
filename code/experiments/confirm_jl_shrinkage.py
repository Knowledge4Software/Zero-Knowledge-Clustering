"""Confirm one development-selected JL predictor on all stored head requests."""
from pathlib import Path
import argparse,json,os,time,csv
import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from commitjl.pilot_io import save,sha
from commitjl.reuse_certificates import integer_stats
from commitjl.jl_certificate_pilots import hint_context,jl_map
from experiments.pilot_jl_certificate_ideas import cases,prediction_labels,distances


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);records=[];started=time.monotonic();seeds=[20261029,20261030,20261031];width=16;alpha=.125;requests=0;pins={}
    save(out/'protocol.json',dict(selection='Every160 stored head update, including36 natural regressions. No outcome filtering. New sketch seeds fixed before confirmation.',development_choice='Rank63 +16 sparse-JL residual measurements with shrinkage1/8; selected as the smallest development candidate improving mean agreement while preserving the99% floor on all19 matched updates/all3 development seeds.',seeds=seeds,width=width,alpha=alpha,scope='Numerical prediction confirmation only. Not a new independent fitted-model cohort. Production model unchanged. No proof-cost or certificate-strength claim.'))
    with threadpool_limits(limits=1):
        for fit,model,x,updates,learned,pin in cases('all'):
            while psutil.virtual_memory().available<800*2**20:save(out/'status.json',dict(complete=False,status='waiting_for_ram',pid=os.getpid(),requests=requests));time.sleep(15)
            pins[fit]=pin;sketches={seed:jl_map(x.shape[1],width,seed) for seed in seeds};h,_=integer_stats(model,x)
            for s,new in updates:
                while psutil.virtual_memory().available<750*2**20:save(out/'status.json',dict(complete=False,status='waiting_for_ram',pid=os.getpid(),requests=requests));time.sleep(15)
                truth=distances(x,new).argmin(1);original=prediction_labels(model,h,new);context=hint_context(model,x,new,[0]*len(x));control=context['central'].argmin(1);base_correct=int((original==truth).sum());control_correct=int((control==truth).sum())
                for seed,sketch in sketches.items():
                    rx=(sketch@context['row_residual'].T).T;rc=(sketch@(context['center_residual']-context['center_residual'][0]).T).T;labels=(context['central']-alpha*2*(rx@rc.T)/3).argmin(1);correct=int((labels==truth).sum());records.append(dict(fit=fit,set=s,seed=seed,n=len(x),original_correct=base_correct,no_jl_gram_correct=control_correct,jl_correct=correct,original_agreement_percent=100*base_correct/len(x),jl_agreement_percent=100*correct/len(x),difference_percentage_points=100*(correct-base_correct)/len(x)))
                requests+=1
                with (out/'results.csv').open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
                save(out/'status.json',dict(complete=False,status='running',pid=os.getpid(),requests=requests,expected_requests=160,active_fit=fit,active_set=s,elapsed_seconds=time.monotonic()-started));print(json.dumps(dict(fit=fit,set=s,requests=requests)),flush=True)
    assert requests==160 and len(records)==480;total=sum(r['n'] for r in records);summary=dict(complete=True,requests=160,sketch_seed_request_pairs=480,native_only=True,original_agreement_percent=100*sum(r['original_correct'] for r in records)/total,no_jl_gram_agreement_percent=100*sum(r['no_jl_gram_correct'] for r in records)/total,jl_agreement_percent=100*sum(r['jl_correct'] for r in records)/total,minimum_original_agreement_percent=min(r['original_agreement_percent'] for r in records),minimum_jl_agreement_percent=min(r['jl_agreement_percent'] for r in records),improved_pairs=sum(r['jl_correct']>r['original_correct'] for r in records),worsened_pairs=sum(r['jl_correct']<r['original_correct'] for r in records),equal_pairs=sum(r['jl_correct']==r['original_correct'] for r in records),elapsed_seconds=time.monotonic()-started,source_hashes=pins)
    save(out/'summary.json',summary);save(out/'status.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='source_hashes'}),flush=True)


if __name__=='__main__':main()
