"""Campaign correctness: pairing, qualification, raw evidence and resource failures."""
from pathlib import Path
from types import SimpleNamespace
import json
import os

import numpy as np
import pandas as pd
import pytest

from experiments import rq2345_complete as suite
from experiments.rq5_precision import measure_precision_row
from commitjl.randomness import CommitRevealChallenge
from commitjl.randomness import TaskLog


def row():
    return dict(dataset=dict(loader='gaussian_mixture',kwargs=dict(n=12,d=6,k=2,gamma=3.,seed=12345)),
                k=2,m=3,q=1,seed=0,clustering_start=0,precision_bits=16,B=4096,backend='groth16',range_bits=20)


def settings():
    return dict(threads=1,proof_threads=2,proof_reps=1,proof_timeout=120,
                proof_rss_gib=2,task_rss_gib=3,task_timeout=30,bootstrap=50,seeds=2,starts=2,min_free_gib=.001)


def test_plan_preserves_axes_and_reports_missing_selection(tmp_path):
    args=SimpleNamespace(stages=suite.STAGES,datasets=['gaussian_easy'],rq1=tmp_path,
                         seeds=10,starts=5,dbscan_points=1000,grinding_trials=256)
    tasks=suite.plan(args)
    assert len({t['id'] for t in tasks})==len(tasks)
    assert any(t['stage']=='rq2_selected' and t.get('unavailable') for t in tasks)
    native=[t for t in tasks if t['stage']=='rq5_native']
    assert all(t['seeds']==10 and t['starts']==5 for t in native)
    assert {t['row']['precision_bits'] for t in tasks if t['stage']=='rq5_precision'}=={8,12,16,24}
    assert all(t['row']['normalization_reference_bits']==24 for t in tasks if t['stage']=='rq5_precision')


def test_commit_reveal_cannot_disclose_before_both_commits():
    challenge=CommitRevealChallenge();challenge.commit('a',bytes(32))
    with pytest.raises(ValueError,match='both parties'): challenge.reveal('a')
    challenge.commit('b',bytes([1])*32)
    assert challenge.reveal('a')==bytes(32)
    assert challenge.finalize()==bytes([1])*32


def test_task_deadline_uses_wall_clock_unless_explicit_clock_is_supplied():
    log=TaskLog(deadline=1.)
    with pytest.raises(PermissionError): log.record_commitment('p',1,2)
    log.record_commitment('p',1,2,now=0.)


def test_certificate_empty_denominator_and_nested_bootstrap():
    result=suite.counts(np.zeros(3,dtype=bool),np.ones(3,dtype=bool))
    assert result['conditional_violation_rate'] is None and result['n_violations']==0
    assert suite.bootstrap_nested(np.ones((3,2)),50)==(1.,1.,1.)
    with pytest.raises(ValueError): suite.bootstrap_nested([[np.nan]],50)


def test_dbscan_artifact_contains_all_pairs_and_core_decisions(tmp_path):
    task=dict(row=row(),max_points=10)
    result=suite.dbscan_measure(task,tmp_path)
    with np.load(tmp_path/'dbscan.npz') as a:
        assert len(a['i'])==45
        assert a['X'].shape==(10,6)
        assert a['original_core'].shape==(10,)
        assert result['n_violations']==0
        original=a['original_sq'];projected=a['projected_sq']
        assert np.array_equal(a['embedding_event'],np.abs(projected-original)<=.15*original+1e-12)
    assert result['original_n']==12


def test_precision_cache_and_raw_artifacts_match_uncached(tmp_path):
    r={**row(),'projection':'sparse_jl','normalization_reference_bits':24}
    cached=measure_precision_row(r,suite.PREREG,cache={},artifact_path=tmp_path/'labels.npz')
    plain=measure_precision_row(r,suite.PREREG)
    for metric in ['ari','r_phi','delta_q95','mean_abs_quantization_error']:
        assert cached[metric]==pytest.approx(plain[metric])
    with np.load(tmp_path/'physical_quantized.npz') as a:
        assert np.max(np.abs(a['Xq']))<=4096
        assert np.allclose(a['decoded'],a['Xq']/2**16)
    assert (tmp_path/'projection_seed0.npz').exists()


def test_phase_qualified_certificates_and_export(tmp_path):
    task=dict(stage='rq4_phase',row={**row(),'projection':'sparse_jl',
              'sweep_factor':'gamma','sweep_value':3.},seeds=2,starts=2,id='phase')
    out=tmp_path/'rq4_phase'/'phase'
    result=suite.worker(task,out,settings());result['status']='ok'
    frame=pd.read_parquet(out/'native_rows.parquet')
    assert len(frame)==4 and frame.n_violations.sum()==0
    assert set(frame.clustering_start)=={0,1} and set(frame.seed)=={0,1}
    suite.write_json(out/'result.json',result)
    suite.write_json(tmp_path/'plan.json',[task]);suite.write_json(tmp_path/'settings.json',settings())
    suite.export(tmp_path)
    summary=pd.read_csv(tmp_path/'summary.csv')
    assert summary.complete.all()
    assert (tmp_path/'figures/phase_qualified_coverage.pdf').exists()
    assert (tmp_path/'all_rows.csv').exists()


def test_same_matrix_proof_ablation_preserves_statement(monkeypatch,tmp_path):
    from commitjl import zk_bridge
    captured=[]
    def fake(instance,**kwargs):
        captured.append(instance)
        return zk_bridge.ProveResult(ok=True,variant=instance.variant,backend=instance.backend,
            num_constraints=123,successful_proofs=1,raw=dict(prove_seconds_samples=[.1],verify_seconds_samples=[.01]))
    monkeypatch.setattr(zk_bridge,'run_prove',fake)
    task=dict(stage='rq3_proof',row=row(),scope='test',id='proof')
    out=tmp_path/'rq3_proof'/'proof';out.mkdir(parents=True)
    result=suite.prove_measure(task,out,settings())
    assert result['status']=='ok' and len(captured)==3
    original,public,unbatched=captured
    for inst in captured:
        assert inst.assign==original.assign and inst.psi_hat==original.psi_hat
        assert inst.c_x==original.c_x and inst.c_theta==original.c_theta
    from commitjl.sparse_gen import sparse_matrix_gen
    assert np.array_equal(public.constant_A,sparse_matrix_gen(original.projection_seed,3,6,1)[1])
    assert public.variant=='prepublished' and unbatched.no_range_batch
    assert (out/'witness.json').exists()
    suite.write_json(out/'result.json',result)
    suite.write_json(tmp_path/'plan.json',[task]);suite.write_json(tmp_path/'settings.json',settings())
    suite.export(tmp_path)
    assert len(pd.read_csv(tmp_path/'proof_summary.csv'))==3
    proof_rows=pd.read_csv(tmp_path/'proof_summary.csv').set_index('arm')
    assert proof_rows.loc['same_matrix_prepublished','variant']=='prepublished'
    assert proof_rows.loc['same_witness_unbatched','no_range_batch']
    assert (tmp_path/'figures/proof_times.pdf').exists()


def test_ablation_export_uses_each_arm_actual_dimensions_and_precision(tmp_path,monkeypatch):
    from commitjl import zk_bridge
    def fake(instance,**kwargs):
        return zk_bridge.ProveResult(ok=True,verify_ok=True,variant=instance.variant,backend=instance.backend,
            num_constraints=123,successful_proofs=1,raw=dict(prove_seconds_samples=[.1]))
    monkeypatch.setattr(zk_bridge,'run_prove',fake)
    r={**row(),'variant':'sparse','precision_bits':24,'normalization_reference_bits':24}
    task=dict(stage='rq5_ablation',row=r,scope='test',id='ablation')
    out=tmp_path/'rq5_ablation/ablation';out.mkdir(parents=True)
    result=suite.prove_measure(task,out,settings())
    suite.write_json(out/'result.json',result)
    suite.write_json(tmp_path/'plan.json',[task]);suite.write_json(tmp_path/'settings.json',settings())
    suite.export(tmp_path)
    rows=pd.read_csv(tmp_path/'proof_summary.csv').set_index('arm')
    assert len(rows)==8
    assert rows.loc['full','variant']=='full' and rows.loc['full','m']==6
    assert rows.loc['reference','m']==3 and rows.loc['reference','precision_bits']==24
    assert rows.loc['lower_precision_16_fixed_reference_24','precision_bits']==16
    assert rows.loc['lower_precision_16_fixed_reference_24','normalization_reference_bits']==24
    assert rows.ari.notna().all() and rows.r_phi.notna().all()


def test_interrupted_later_arm_preserves_completed_proof_and_failed_task(tmp_path,monkeypatch):
    from commitjl import zk_bridge
    calls=[]
    def prove(instance,**kwargs):
        calls.append(instance)
        if len(calls)>1: raise RuntimeError('later arm interrupted')
        return zk_bridge.ProveResult(ok=True,verify_ok=True,variant=instance.variant,backend=instance.backend,
            num_constraints=123,successful_proofs=1,raw=dict(prove_seconds_samples=[.1]))
    monkeypatch.setattr(zk_bridge,'run_prove',prove)
    task=dict(stage='rq3_proof',row=row(),scope='test',id='partial')
    out=tmp_path/'rq3_proof/partial';out.mkdir(parents=True)
    with pytest.raises(RuntimeError,match='later arm'):
        suite.prove_measure(task,out,settings())
    assert len(json.loads((out/'proof_arms.json').read_text()))==1
    monkeypatch.setattr(suite.subprocess,'Popen',lambda *a,**k:SimpleNamespace(returncode=1,poll=lambda:1))
    result=suite.execute(task,tmp_path,settings())
    assert result['status']=='failed' and result['completed_proof_arms']==1
    assert result['error']=='previous_task_interrupted'
    assert result['arms'][0]['arm']=='reference' and result['arms'][0]['ok']
    suite.write_json(tmp_path/'plan.json',[task]);suite.write_json(tmp_path/'settings.json',settings())
    suite.export(tmp_path)
    assert pd.read_csv(tmp_path/'inventory.csv').status.tolist()==['failed']
    exported=pd.read_csv(tmp_path/'proof_summary.csv')
    assert exported.arm.tolist()==['reference'] and exported.prove_median.iloc[0]==pytest.approx(.1)
    assert exported.task_status.tolist()==['failed']


def test_parent_interrupt_records_failure_and_recovered_proof_checkpoint(tmp_path,monkeypatch):
    import psutil
    task=dict(stage='rq3_proof',row=row(),scope='test',id='interrupt')
    out=tmp_path/'rq3_proof/interrupt';out.mkdir(parents=True)
    proc=SimpleNamespace(pid=123,returncode=None,poll=lambda:None)
    def interrupt(*args,**kwargs):
        suite.write_json(out/'proof_arms.json',[dict(arm='reference',ok=True)])
        raise KeyboardInterrupt()
    monkeypatch.setattr(suite.subprocess,'Popen',lambda *a,**k:proc)
    monkeypatch.setattr(psutil,'Process',interrupt)
    monkeypatch.setattr(suite,'kill_tree',lambda p:setattr(p,'returncode',-9))
    with pytest.raises(KeyboardInterrupt): suite.execute(task,tmp_path,settings())
    result=json.loads((out/'result.json').read_text())
    assert result['status']=='failed' and result['error']=='interrupted'
    assert result['completed_proof_arms']==1 and result['child_exit_code']==-9
    assert suite.execute(task,tmp_path,settings())==result


def test_resource_kill_is_saved_and_resume_does_not_rerun(tmp_path):
    task=dict(stage='rq4_dbscan',row=row(),max_points=10,id='limit')
    s={**settings(),'task_rss_gib':.000001}
    suite.write_json(tmp_path/'settings.json',s)
    result=suite.execute(task,tmp_path,s)
    assert result['status']=='failed' and result['error']=='memory_limit'
    assert result['child_exit_code']!=0
    assert suite.execute(task,tmp_path,s)==result


def test_ordering_times_exact_same_matrix_and_saves_substages(tmp_path):
    result=suite.ordering_measure(dict(row=row(),source='commit_reveal'),tmp_path)
    samples=pd.DataFrame(json.loads((tmp_path/'timings.json').read_text()))
    assert len(samples)==20
    assert set(samples.arm)=={'postcommit','prepublished'}
    assert samples[samples.arm=='postcommit'].substage_hash_CA.notna().all()
    assert result['measured_pairs']==10 and result['scope'].endswith('replayed_challenge')


def test_frozen_data_hash_and_missing_quality_gate(tmp_path):
    from experiments.pipeline import load_row_dataset
    data=load_row_dataset(row())
    frozen=tmp_path/'rq1'/'gaussian_easy'
    suite.save_npz(frozen/'artifacts/dataset.npz',X=data['X'])
    suite.write_json(frozen/'manifest.json',dict(dataset=data['meta']))
    suite.write_json(frozen/'completion.json',dict(rows=1))
    assert np.array_equal(suite.frozen_dataset(frozen)['X'],data['X'])
    out=tmp_path/'campaign'/'rq2_selected'/'test';out.mkdir(parents=True)
    result=suite.prove_measure(dict(stage='rq2_selected',row={**row(),'variant':'sparse'},
        name='gaussian_easy',directory=str(frozen),scope='test'),out,settings())
    assert result['status']=='unavailable'
    limited=suite.prove_measure(dict(stage='rq2_selected',row={**row(),'variant':'sparse'},
        name='gaussian_easy',directory=str(frozen),scope='test'),out,{**settings(),'max_proof_input_elements':8})
    assert limited['status']=='excluded' and limited['input_elements']==72
    assert limited['failure_kind']=='preflight_budget_exclusion_not_measured_OOM'
    suite.save_npz(frozen/'artifacts/dataset.npz',X=data['X']+1)
    with pytest.raises(ValueError,match='hash'): suite.frozen_dataset(frozen)


def test_json_records_nonfinite_measurements_as_null(tmp_path):
    suite.write_json(tmp_path/'row.json',dict(ratio=float('nan'),count=np.int64(3)))
    assert json.loads((tmp_path/'row.json').read_text())==dict(ratio=None,count=3)


def test_quality_maps_replay_the_commitment_challenge_and_exact_assignments(tmp_path):
    from experiments.pipeline import load_row_dataset
    from commitjl.protocol import derive_seed
    from commitjl.sparse_gen import sparse_matrix_gen
    from commitjl.clustering import nearest_centroid_assign
    samples=suite.quantized_quality_samples({**row(),'variant':'sparse'},load_row_dataset(row()),2,2,tmp_path)
    assert len(samples)==4
    with np.load(tmp_path/'quality_input.npz') as input_file:
        Xq=input_file['Xq']
        for r in samples:
            seed=derive_seed(int(r['c_x']),int(r['c_theta']),bytes.fromhex(r['rho_hex']))
            assert seed.hex()==r['projection_seed_hex']
            A=sparse_matrix_gen(seed,3,6,1)[1]
            with np.load(tmp_path/f"baseline_start{r['clustering_start']}.npz") as baseline, np.load(
                tmp_path/f"map_seed{r['seed']}_start{r['clustering_start']}.npz") as artifact:
                assert np.array_equal(A,artifact['A'])
                assert np.array_equal(artifact['labels'],nearest_centroid_assign(Xq@A.T,baseline['Cq']@A.T))


def test_mixed_campaign_curves_include_dbscan_and_connect_k_q_sweeps(tmp_path,monkeypatch):
    import matplotlib.axes
    recorded=[];original=matplotlib.axes.Axes.plot
    def record(ax,x,y,*args,**kwargs):
        recorded.append((ax, list(x),kwargs.get('label','')))
        return original(ax,x,y,*args,**kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes,'plot',record)
    rows=[]
    for m in [32,64]:
        rows.append(dict(stage='rq4_dbscan',name='dbscan_case',m=m,q=1,k=np.nan,
            qualified_coverage_mean=.5,qualified_coverage_ci_low=.4,qualified_coverage_ci_high=.6))
    for k in [2,3,5]:
        rows.append(dict(stage='rq5_native',name='synthetic',m=64,q=2,k=k,sweep_factor='k',sweep_value=k,
            ari_mean=.8,ari_ci_low=.7,ari_ci_high=.9))
    for q in [1,2,4]:
        rows.append(dict(stage='rq5_native',name='synthetic',m=64,q=q,k=5,sweep_factor='q',sweep_value=q,
            ari_mean=.8,ari_ci_low=.7,ari_ci_high=.9))
    suite.plot_exports(tmp_path,pd.DataFrame(rows),[])
    assert any(len(x)==2 and label.startswith('dbscan_case') for _,x,label in recorded)
    assert any(x==[2,3,5] for _,x,_ in recorded)
    assert any(x==[1,2,4] for _,x,_ in recorded)
    assert (tmp_path/'figures/rq4_dbscan_m_qualified_coverage.pdf').exists()


def test_scaling_does_not_certify_grid_with_missing_level(tmp_path):
    tasks=[dict(stage='rq5_native',row=dict(sweep_factor='n',sweep_value=n))
           for n in [8,16,32,64]]
    suite.write_json(tmp_path/'plan.json',tasks)
    suite.write_json(tmp_path/'settings.json',settings())
    pd.DataFrame([dict(stage='rq5_native',status='ok')]).to_csv(tmp_path/'inventory.csv',index=False)
    rows=[dict(stage='rq5_native',status='ok',sweep_factor='n',sweep_value=n,
               seed=seed,clustering_start=start,t_assign_s=n*.01,t_proj_build_s=n*.02,t_kmeans_s=n*.03)
          for n in [8,16,32] for seed in range(2) for start in range(2)]
    suite.scaling_exports(tmp_path,pd.DataFrame(rows))
    slopes=pd.read_csv(tmp_path/'scaling_slopes.csv')
    assert not slopes.complete.any()
    assert 'slope' not in slopes or slopes.slope.isna().all()


def test_failed_child_cannot_keep_success_result(tmp_path,monkeypatch):
    task=dict(stage='rq4_dbscan',id='stale',row=row())
    def fake_process(*args,**kwargs):
        suite.write_json(tmp_path/'rq4_dbscan/stale/result.json',dict(status='ok',rows=1))
        return SimpleNamespace(returncode=1,poll=lambda:1)
    monkeypatch.setattr(suite.subprocess,'Popen',fake_process)
    result=suite.execute(task,tmp_path,settings())
    assert result['status']=='failed' and result['error']=='child_process_failed'
    assert result['child_exit_code']==1


@pytest.mark.skipif(os.environ.get('COMMITJL_LIVE_SUITE_TEST')!='1',reason='explicit real gnark smoke')
def test_live_proof_arms(tmp_path):
    r=row();r['dataset']['kwargs'].update(n=6,d=4);r.update(m=2)
    result=suite.prove_measure(dict(stage='rq3_proof',row=r,scope='live_smoke'),tmp_path,settings())
    assert result['status']=='ok',result
    assert all(a['successful_proofs']==1 for a in result['arms'])
    assert len(json.loads((tmp_path/'proof_arms.json').read_text()))==3
