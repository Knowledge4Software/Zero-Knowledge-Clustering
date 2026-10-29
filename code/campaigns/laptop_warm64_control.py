"""Actual64-request warm comparison; authenticated data/setup reused for both.

Every request proof is generated afresh. Reused row evidence and proving keys
are explicitly distinguished from newly timed work. No matched Full64 ratio.
"""
from pathlib import Path
import copy
import json
import shutil
import sys
import time

from campaigns.laptop_four_hour_research import BASIC,PRIOR,DeadlineCampaign
from campaigns.published_baselines_laptop import GO,RUST,read,save,sha


def run(root,deadline):
    import numpy as np
    from validation.check_group_fast_proofs import export_public
    from commitjl.group_change_bounds import exact_old_correction_bounds
    from commitjl.group_moment_reference import prepare_groups
    from experiments.pilot_jl_certificate_ideas import distances

    root.mkdir(parents=True,exist_ok=False)
    source=BASIC/'matched/fashion_n32'
    data=np.load(source/'ordered_head_diagnostic_inputs.npz')
    centers=data['centers'].tolist()
    assert len(centers)==65 and len({json.dumps(c) for c in centers})==65
    save(root/'plan.json',dict(
        requests=64,n=32,source_archive_sha256=sha(source/'ordered_head_diagnostic_inputs.npz'),
        source_basic_plan_sha256=sha(BASIC/'plan.json'),deadline_unix=deadline,
        comparison='64 freshly generated request proofs per system after shared data/key preparation. New candidates previously frozen; all retained.',
        reused_certificate='Authenticated row evidence, commitment and three base keys; refinement key reused when available.',
        reused_exact='Halo2 parameters, proving/verification keys and old-reference cost proof.',
        phase_seconds=3600,workers=8,rss_cap_mib=3584,
        no_full64_reference=True,one_exploratory_timing_execution=True))
    save(root/'source_pin.json',dict(plan_sha256=sha(root/'plan.json'),
        warm_runner_sha256=sha(Path(__file__)),
        refinement_runner_sha256=sha(Path(__file__).with_name('run_group_refinement_control.py')),
        go_binary_sha256=sha(GO),rust_binary_sha256=sha(RUST),
        original_owner_pin_sha256=sha(source/'ordered_head_base/owner_pin.json'),
        original_exact_owner_pin_sha256=sha(source/'zkml_owner_pin.json')))
    campaign=DeadlineCampaign(root,deadline)
    parent=root/'parent';base=root/'base';base.mkdir()
    prediction=read(source/'ordered_head/head_r63/input.json')
    prediction['reuse_centers']=centers
    save(parent/'head_r63/input.json',prediction)
    full=copy.deepcopy(prediction);full['model'].update(method='full',basis=[],seed='')
    full['queries']=[list(range(i,i+8)) for i in range(0,32,8)]
    save(parent/'full/input.json',full)
    protocol=read(source/'ordered_head/protocol.json')
    protocol.update(fitting_seconds=0.,phase_wall_limit_seconds=3600)
    save(parent/'protocol.json',protocol)
    source_base=source/'ordered_head_base'
    # Copy reusable evidence only. All query proofs and query caches are omitted,
    # so this is a fresh warm64 generation run, not cached workload timing.
    def skip_query(directory,names):
        return [name for name in names if name.startswith('group_query')]
    shutil.copytree(source_base/'bundle',base/'bundle',ignore=skip_query)
    shutil.copyfile(source_base/'owner_pin.json',base/'owner_pin.json')
    inp=read(source_base/'input.json');inp['reuse_centers']=centers
    save(base/'input.json',inp)
    task=dict(reuse_centers=centers[1:],queries=inp['queries'],comparison_base=centers[0])
    save(base/'approved.json',task)
    metadata=read(source_base/'protocol.json')
    metadata.update(source_sha256=sha(parent/'head_r63/input.json'),fitting_seconds=0.,
        proposal_source_indices=list(range(1,65)),scope='All64 fixed distinct public updates; no outcome filtering.',
        warm_setup_reuse=True)
    save(base/'protocol.json',metadata)
    resources={}
    for phase in ['group_compact_prepare','group_keys','group_rows']:
        prior=read(source_base/'status.json')['resources'][phase]
        resources[phase]=dict(complete=True,wall_seconds=0.,reused_evidence=True,
            source=str(source_base/'status.json'),source_phase=phase,
            original_wall_seconds=prior['wall_seconds'],new_generation=False)
    save(base/'status.json',dict(complete=False,resources=resources,status='running'))

    def go_phase(label,phase,out):
        return campaign.invoke(label,[GO,'-phase',phase,'-instance',base/'input.json',
            '-out',out,'-q','16','-reps','1','-warmup','0','-max-constraints','1600000',
            '-trust',base/'owner_pin.json','-public-task',base/'approved.json'],3600)

    resources['group_query']=go_phase('certificate_query64','group_query',base/'bundle')
    export_public(base/'bundle',base/'public')
    resources['group_verify']=go_phase('certificate_verify64','group_verify',base/'public')
    x=np.asarray(inp['x'],np.int64);old=np.asarray(centers[0],np.int64)
    od=distances(x,old);groups=od.argmin(1);prepared=prepare_groups(x,groups.tolist(),5)
    old_cost=int(od.min(1).sum());exact_costs=[];native=[]
    verified=read(base/'public/group_verified_decisions.json')
    assert len(verified)==64
    for i,(new,decision) in enumerate(zip(centers[1:],verified)):
        cost=int(distances(x,np.asarray(new,np.int64)).min(1).sum());exact_costs.append(cost)
        truth=cost-old_cost
        b=exact_old_correction_bounds(inp['model'],x,prepared,old,new)
        assert int(decision['lower'])==b['change_lower_scaled']
        assert int(decision['upper'])==b['change_upper_scaled']
        assert int(decision['lower'])<=truth*int(decision['denominator'])<=int(decision['upper'])
        native.append(dict(set=i,source_set=i+1,truth=truth))
    save(base/'native_checks.json',dict(complete=True,decisions=native,all_intervals_contain_truth=True))
    save(base/'status.json',dict(complete=True,resources=resources,status='complete'))
    refined=root/'refined';refined.mkdir()
    old_refined=source/'ordered_head_refined'
    if (old_refined/'bundle/trusted.json').exists():
        # Reuse only refinement keys/setup; no correction proofs are copied.
        def skip_refinement(directory,names):
            return [name for name in names if name.startswith('refinement')]
        shutil.copytree(old_refined/'bundle',refined/'bundle',ignore=skip_refinement)
    # The command creates its protocol afresh; a precreated empty directory is
    # permitted with --resume only after protocol creation, so use an unused path
    # for the actual command and move the reusable key directory into it below.
    refinement_keys=refined/'bundle'
    if refinement_keys.exists():
        keys=root/'reusable_refinement_keys';refinement_keys.rename(keys)
    else:keys=None
    refined.rmdir()
    if keys is not None:
        # Existing runner accepts a newly created directory only when resume is
        # false; stage key reuse with an explicit new optional CLI argument.
        key_args=['--reuse-key-bundle',str(keys)]
    else:key_args=[]
    refinement_receipt=campaign.invoke('certificate_refine64',[sys.executable,'-m','campaigns.run_group_refinement_control',
        '--base',base,'--parent',parent,'--root',refined,'--binary',GO,
        '--smoke-receipt',PRIOR/'refinement_smoke/checks.json','--no-full-reference',*key_args],3600)
    check=read(refined/'checked_comparison.json')
    assert check['complete'] and len(check['decisions'])==64

    # Exact path: same original rows/salt,64 new center banks, same reusable
    # shape/key and owner pin. Old cost proof is replayed separately, not remade.
    exact_input=copy.deepcopy(full);exact_input['reuse_centers']=centers[1:]
    save(root/'exact_input.json',exact_input)
    save(root/'approved_exact.json',dict(reuse_centers=centers[1:]))
    private=root/'exact_private';private.mkdir()
    for name in ['params','pk','vk','owner_pin.json']:
        shutil.copyfile(source/'zkml_private'/name,private/name)
    public=root/'exact_public';public.mkdir()
    for name in ['params','vk']:shutil.copyfile(private/name,public/name)
    shutil.copyfile(private/'owner_pin.json',root/'exact_owner_pin.json')
    old_public=root/'old_reference_public';old_public.mkdir()
    for name in ['params','vk']:shutil.copyfile(private/name,old_public/name)
    shutil.copytree(source/'zkml_public/bank_000',old_public/'bank_000')
    assert read(old_public/'bank_000/public.json')['cost']==old_cost
    old_receipt=campaign.invoke('exact_old_reference_replay',[RUST,'verify',old_public,0,
        root/'exact_owner_pin.json',source/'approved_task.json'],300)
    query_receipt=campaign.invoke('exact_query64',[RUST,'queries',root/'exact_input.json',private],3600)
    replay_seconds=old_receipt['wall_seconds'];proof_bytes=0
    for i,cost in enumerate(exact_costs):
        folder=f'bank_{i:03}';(public/folder).mkdir()
        for name in ['proof','public.json']:shutil.copyfile(private/folder/name,public/folder/name)
        value=read(private/folder/'result.json')
        assert value['cost']==cost;proof_bytes+=value['proof_bytes']
        receipt=campaign.invoke(f'exact_verify{i:03}',[RUST,'verify',public,i,
            root/'exact_owner_pin.json',root/'approved_exact.json'],300)
        replay_seconds+=receipt['wall_seconds']
    certificate_seconds=(resources['group_query']['wall_seconds']+resources['group_verify']['wall_seconds']
        +refinement_receipt['wall_seconds'])
    exact_seconds=query_receipt['wall_seconds']+replay_seconds
    result=dict(complete=True,requests=64,n=32,method='centroid certificate',
        measured_warm_certificate_seconds=certificate_seconds,measured_warm_exact_seconds=exact_seconds,
        exact_over_certificate=exact_seconds/certificate_seconds,
        fallback_updates=check['fallback_updates'],conclusive_before_fallback=check['conclusive_without_fallback'],
        fresh_refinement_proofs=check['new_refinement_proofs'],exact_proofs=64,exact_proof_bytes=proof_bytes,
        certificate_exported_proof_bytes=sum(p.stat().st_size for directory in [base/'public',refined/'public',refined/'fallback/full/public_bundle'] if directory.exists() for p in directory.rglob('*.proof')),
        exact_exported_proof_bytes_including_old=proof_bytes+(old_public/'bank_000/proof').stat().st_size,
        all_intervals_contain_truth=True,no_wrong_decisions=True,setup_reused_for_both=True,
        public_only_replay=True,old_cost=old_cost,exact_costs=exact_costs,
        timing='Fresh64-request warm proof/query/replay phase totals. Shared setup/row generation excluded for both; cached evidence is replayed. Correction/fallback phases charged. Export copying and independent oracle checks are outside these phase totals. One exploratory execution.',
        public_manifest={p.relative_to(root).as_posix():sha(p) for directory in [public,old_public,base/'public',refined/'public'] if directory.exists() for p in directory.rglob('*') if p.is_file()})
    save(root/'checked_comparison.json',result)
    campaign.state.update(complete=True,status='complete',active_phase=None,finished_unix=time.time())
    save(root/'matched_queue.json',campaign.state)
    print(json.dumps(result,indent=2),flush=True)
