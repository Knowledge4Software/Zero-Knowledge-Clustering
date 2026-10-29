"""Paired exact-output native kernel benchmark, never a proof speed claim."""
from pathlib import Path
import argparse,json,subprocess,time,types
import numpy as np
from campaigns.local_fast_tradeoff import save,sha
from commitjl.reuse_certificates import integer_stats
from commitjl.paired_certificates import change_parameters
from commitjl.group_change_bounds import exact_old_correction_bounds
from commitjl.group_moment_reference import prepare_groups

REPO=Path(__file__).resolve().parents[2]


def previous(name):
    source=subprocess.check_output(['git','show','85da2c1:code/commitjl/'+name+'.py'],cwd=REPO).decode()
    module=types.ModuleType('commitjl._previous_'+name);module.__package__='commitjl';exec(compile(source,name,'exec'),module.__dict__)
    return module


def run(out):
    if out.exists():raise RuntimeError('Preserve previous measurements')
    old_stats=previous('reuse_certificates').integer_stats
    old_params=previous('paired_certificates');old_params.integer_stats=old_stats
    old_bounds=previous('group_change_bounds');old_bounds.integer_stats=old_stats;old_bounds.change_parameters=old_params.change_parameters
    source=REPO/'code/validation/outputs/referee_final_20261025/inputs/mnist_switching_mature_n128/ordered_head'
    inp=json.loads((source/'head_r63/input.json').read_text());aux=json.loads((source/'auxiliary_basis.json').read_text())
    model=dict(inp['model'],basis=aux['certificate_basis']);rows=np.asarray(inp['x'],dtype=np.int64)
    old=np.asarray(model['centers'],dtype=np.int64);new=np.asarray(inp['reuse_centers'][1],dtype=np.int64)
    dist=((rows[:,None,:]-old[None,:,:])**2).sum(2);prepared=prepare_groups(rows,dist.argmin(1).tolist(),len(old))
    results=[]
    for name,baseline,optimized,args in [
        ('integer_stats',old_stats,integer_stats,(model,rows)),
        ('center_parameters',old_params.change_parameters,change_parameters,(model,new,new)),
        ('complete_native_bounds',old_bounds.exact_old_correction_bounds,exact_old_correction_bounds,(model,rows,prepared,old,new))]:
        def normalize(value):
            if isinstance(value,np.ndarray):return normalize(value.tolist())
            if isinstance(value,dict):return {k:normalize(v) for k,v in value.items()}
            if isinstance(value,(tuple,list)):return [normalize(v) for v in value]
            if isinstance(value,np.integer):return int(value)
            return value
        assert normalize(baseline(*args))==normalize(optimized(*args)),name+' changed exact outputs'
        times={'baseline':[],'optimized':[]}
        for repetition in range(3):
            order=[('baseline',baseline),('optimized',optimized)]
            if repetition%2:order.reverse()
            for label,function in order:
                tick=time.perf_counter();function(*args);times[label].append(time.perf_counter()-tick)
        results.append(dict(kernel=name,identical_exact_outputs=True,seconds=times,
            median_ratio=float(np.median(times['baseline'])/np.median(times['optimized']))))
    save(out/'checks.json',dict(complete=True,scope='Native integer kernels only. This is not a proving or deployment speed ratio.',
        baseline_commit='85da2c1',input_sha256=sha(source/'head_r63/input.json'),auxiliary_sha256=sha(source/'auxiliary_basis.json'),n=len(rows),results=results))
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out.resolve())
