"""Serialized-proof verifier regressions; intentionally use no private rows."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

CODE=Path(__file__).resolve().parents[1]
EXE=Path(os.environ.get('COMMITJL_BOUND_BIN',CODE/'zk/bin/commitjl-bound.exe'))


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    if not EXE.is_file():pytest.skip('Build commitjl-bound first')
    root=tmp_path_factory.mktemp('bound-proof-bundle')
    model=dict(version='bound-audit-v1',method='protected_tail',n=4,d=4,k=2,
        input_bits=4,basis_scale=8,input_scale='1',centers=[[0]*4,[2,0,0,0]],
        basis=[[8,0,0,0]],tail_rows=[0,1,0,1],tail_signs=[1,-1,1,1],tail_width=2,seed='public-fixture')
    data=dict(model=model,x=[[1,1,0,0],[3,2,0,1],[-2,1,1,0],[2,-1,0,2]],
              salt_x='123',salt_z='456',queries=[[0,2],[1,3]])
    inp=root/'input.json';inp.write_text(json.dumps(data));out=root/'bundle'
    env=dict(os.environ,GOMAXPROCS='2',GOMEMLIMIT='1GiB')
    for phase in ['prepare','stage1','query','geometry']:
        subprocess.run([str(EXE),'-phase',phase,'-instance',str(inp),'-out',str(out),
            '-q','2','-reps','1','-warmup','0'],check=True,capture_output=True,env=env,timeout=90)
    # A verifier must work after private input and prepared rows are removed.
    inp.unlink();(out/'prepared.gob').unlink();(out/'statistics.json').unlink()
    return out


def verify(root,expected):
    p=subprocess.run([str(EXE),'-phase','verify','-out',str(root),'-trust',str(root/'trusted.json')],
        capture_output=True,text=True,timeout=30)
    assert (p.returncode==0)==expected,(p.stdout,p.stderr)
    if expected:assert json.loads((root/'verify.json').read_text())['verified']==8


def test_verifier_needs_only_public_statements_and_proofs(bundle):
    verify(bundle,True)


@pytest.mark.parametrize('attack',['missing_row','duplicate_index','wrong_cost','key_substitution','different_original_root','wrong_geometry_distance','geometry_pair_substitution'])
def test_verifier_rejects_incomplete_or_substituted_bundles(bundle,tmp_path,attack):
    out=tmp_path/'tampered';shutil.copytree(bundle,out)
    if attack=='missing_row':(out/'stage1/row_000001.proof').unlink()
    elif attack=='key_substitution':shutil.copyfile(out/'keys/query_2.vk',out/'keys/stage1.vk')
    else:
        file=out/('stage1.json' if attack=='duplicate_index' else 'query_2.json' if attack=='wrong_cost' else 'geometry.json' if attack.startswith(('wrong_geometry','geometry_pair')) else 'prepare.json')
        data=json.loads(file.read_text())
        if attack=='duplicate_index':data['samples'][1]['id']=0
        elif attack=='wrong_cost':data['samples'][0]['psi']=str(int(data['samples'][0]['psi'])+1)
        elif attack=='wrong_geometry_distance':data['samples'][0]['psi']=str(int(data['samples'][0]['psi'])+1)
        elif attack=='geometry_pair_substitution':data['samples'][0]['indices']=[0,3]
        else:data['statement']['cx']=str(int(data['statement']['cx'])+1)
        file.write_text(json.dumps(data))
    verify(out,False)
