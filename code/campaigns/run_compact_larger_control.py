"""Predeclared 256-record natural/near-identity stress control with fresh Full."""
from pathlib import Path
import argparse,json,copy,subprocess,sys,os
import numpy as np
from campaigns.local_fast_tradeoff import save,sha,run as run_full
from experiments.review_update_decisions import array_sha
from data.loaders import load_dataset
from analysis.audit_local_fast_tradeoff import audit_case
from commitjl.reuse_certificates import certificate


def build(root):
    repo=Path(__file__).resolve().parents[2];source=repo/'paper_results/review_controls/local_group_moments_extended_20261023';name='fashion_mnist_k3_seed11_age8_learned';frozen=np.load(source/(name+'.npz'));pin=next(x for x in json.loads((source/'pins.json').read_text()) if x['fit']==name)
    loaded=load_dataset('fashion_mnist',n_max=5000,seed=0);x=np.asarray(loaded['X'],float);x=x[np.linalg.norm(x,axis=1)>0];indices=np.random.default_rng(741+11).permutation(len(x));cut=int(.6*len(x));training=x[indices[:cut]];scale=4095/max(float(np.abs(training).max()),1e-12)
    all_rows=np.clip(np.rint(x[indices[cut:cut+1024]]*scale),-4096,4096).astype(np.int64);assert array_sha(all_rows)==pin['audit_sha256'];assert array_sha(np.rint(training*scale).astype(np.int64))==pin['training_sha256']
    rows=all_rows[:256];old=frozen['centers'][0].tolist();proposals=frozen['centers'][1:].tolist();rng=np.random.default_rng(20261025);directions=rng.choice([-1,1],size=np.asarray(old).shape)
    for step in [-2,-1,1,2]:proposals.append(np.clip(np.asarray(old)+step*directions,-4096,4096).tolist())
    assert len({json.dumps(c) for c in proposals})==8 and old not in proposals
    model=dict(version='bound-audit-v1',method='learned_energy',n=256,d=784,k=3,input_bits=12,basis_scale=16384,input_scale=str(scale),centers=old,basis=frozen['basis'].tolist(),tail_width=0,tail_rows=[],tail_signs=[],seed='frozen historical fit; no refit')
    inp=dict(model=model,x=rows.tolist(),salt_x='20261025',salt_z='20261026',queries=[list(range(i,i+16)) for i in range(0,256,16)],reuse_centers=[old]+proposals,comparison_base=old)
    parent=root/'parent';save(parent/'head_r63/input.json',inp);full=copy.deepcopy(inp);full['model'].update(method='full',basis=[],seed='');full['queries']=[list(range(i,i+8)) for i in range(0,256,8)];save(parent/'full/input.json',full)
    checks=[]
    for i,c in enumerate(proposals):
        truth=((rows[:,None,:]-np.asarray(c)[None,:,:])**2).sum(2).argmin(1);labels=certificate(model,rows,c)['labels'];checks.append(dict(set=i,kind='natural fitted update' if i<4 else 'predeclared near-identity diagnostic',agreement_percent=100*float(np.mean(truth==labels))))
    save(parent/'protocol.json',dict(fitting_seconds=0.,fitting_cost_scope='Frozen supplied model; fitting not measured in this scale control.',source_fit=name,source_archive_sha256=sha(source/(name+'.npz')),source_pins_sha256=sha(source/'pins.json'),selection='First256 rows of the verified original1024 audit catalog, all4 actual updates, plus all4 predeclared symmetric perturbations. No outcome/accuracy filtering.',prediction_checks=checks,quantizer=scale))
    return parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','binary','smoke-receipt']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--certificate-scale-bits',type=int,default=14);p.add_argument('--resume',action='store_true');a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=a.resume)
    parent=root/'parent' if a.resume else build(root)
    # Full gets fresh setup, generation and public replay; baseline replays are
    # never counted as new fallback generation or new baseline generation.
    run_full(parent,a.binary.resolve(),['full']);baseline,_,_=audit_case(parent,'full');save(root/'full_reference.json',baseline)
    command=[sys.executable,'-m','campaigns.run_group_fast_control','--compact','--certificate-scale-bits',str(a.certificate_scale_bits),'--parent',str(parent),'--root',str(root/'compact'),'--binary',str(a.binary.resolve()),'--smoke-receipt',str(a.smoke_receipt.resolve())]
    if (root/'compact').exists():command.append('--resume')
    subprocess.run(command,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    result=json.loads((root/'compact/checked_comparison.json').read_text());save(root/'checked_comparison.json',result)


if __name__=='__main__':main()
