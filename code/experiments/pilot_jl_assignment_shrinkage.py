"""Small prediction-only JL residual correction screen, all matched updates.

Fixed shrinkage grid, all three seeds retained. No production model changes,
no certified acceptance from JL estimates, no proof-speed claim.
"""
from pathlib import Path
import argparse,time,os,json,csv
import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from commitjl.pilot_io import save,sha
from commitjl.reuse_certificates import integer_stats
from commitjl.jl_certificate_pilots import hint_context,jl_map


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);repo=Path(__file__).resolve().parents[2];source=repo/'code/validation/outputs/local_fast_tradeoff_n128_20261024/head_r63/input.json';inp=json.loads(source.read_text());x=np.asarray(inp['x'],dtype=np.int64);records=[];started=time.monotonic();seeds=[20261026,20261027,20261028];alphas=[0.,.125,.25,.5,1.];widths=[8,16,32,64]
    save(out/'protocol.json',dict(input_sha256=sha(source),selection='All19 original distinct nonidentical N128 updates. Fixed ranks39/63, four sketch sizes, five shrinkage coefficients and three fixed seeds. No outcome filtering.',ranks=[39,63],widths=widths,alphas=alphas,seeds=seeds,scope='Exploratory prediction-only grid. Original production prediction model remains unchanged. Alpha0 is an exact-Gram corrected no-JL control; compare against it before attributing benefit to JL. Feature count is not measured proof cost. Passing candidates require >=99% agreement on every update for every seed and further confirmation.'))
    with threadpool_limits(limits=1):
        sketches={(m,s):jl_map(x.shape[1],m,s) for m in widths for s in seeds}
        for rank in [39,63]:
            model=dict(inp['model'],basis=inp['model']['basis'][:rank]);h,e=integer_stats(model,x)
            for index,centers in enumerate(inp['reuse_centers']):
                if centers==inp['model']['centers']:continue
                while psutil.virtual_memory().available<750*2**20:save(out/'status.json',dict(complete=False,status='waiting_for_ram',pid=os.getpid(),rank=rank,set=index));time.sleep(15)
                c=np.asarray(centers,dtype=np.int64);truth=((x[:,None,:]-c[None,:,:])**2).sum(2).argmin(1);context=hint_context(model,x,c,[0]*len(x));hc,ec=integer_stats(model,c);constant=np.asarray([sum(int(v)**2 for v in row)+int(v) for row,v in zip(hc,ec)],dtype=object);original=(-2*h.astype(object)@hc.astype(object).T+constant).argmin(1)
                records.append(dict(rank=rank,set=index,width=0,seed=0,alpha=0.,kind='original_head_energy',agreement_percent=100*float(np.mean(original==truth)),stored_features=rank+1))
                for (width,seed),sketch in sketches.items():
                    rx=(sketch@context['row_residual'].T).T;rc=(sketch@(context['center_residual']-context['center_residual'][0]).T).T;correction=2*(rx@rc.T)/3
                    for alpha in alphas:
                        labels=(context['central']-alpha*correction).argmin(1);records.append(dict(rank=rank,set=index,width=width,seed=seed,alpha=alpha,kind='jl_shrinkage',agreement_percent=100*float(np.mean(labels==truth)),stored_features=rank+1+width))
                keys=list(records[0]);
                with (out/'results.csv').open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(records)
                save(out/'status.json',dict(complete=False,status='running',pid=os.getpid(),rank=rank,set=index,records=len(records),elapsed_seconds=time.monotonic()-started));print(json.dumps(dict(rank=rank,set=index)),flush=True)
    summary=[]
    for rank in [39,63]:
        original=[r['agreement_percent'] for r in records if r['rank']==rank and r['kind']=='original_head_energy'];summary.append(dict(rank=rank,kind='original_head_energy',minimum_agreement=min(original),mean_agreement=sum(original)/len(original)))
        for width in widths:
            for alpha in alphas:
                group=[r for r in records if r['rank']==rank and r['kind']=='jl_shrinkage' and r['width']==width and r['alpha']==alpha];assert len(group)==57;values=[r['agreement_percent'] for r in group];summary.append(dict(rank=rank,width=width,alpha=alpha,kind='jl_shrinkage',minimum_agreement=min(values),mean_agreement=sum(values)/len(values),passes_all_updates_all_seeds=min(values)>=99.,stored_features=rank+1+width))
    result=dict(complete=True,native_only=True,production_prediction_unchanged=True,unique_updates=19,elapsed_seconds=time.monotonic()-started,results=summary);save(out/'summary.json',result);save(out/'status.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
