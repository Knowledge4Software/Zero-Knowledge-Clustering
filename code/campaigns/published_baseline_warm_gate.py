"""Equal cost-aware routing for every frozen representation in the local gate.

This engineering follow-up reuses the completed cold setups but generates all
changing-request proofs afresh. Calibration comes from the earlier seed521 Q8
pilot, equally for all bases, not from the new cohort's observed request times.
"""
import argparse
import datetime
import os
from pathlib import Path
import time

from campaigns.published_baselines_laptop import Campaign,GO,read,save,sha
from campaigns.cost_aware_pilot import calibrate,prove_case
from campaigns.laptop_advantage_pilot import DEFAULT as CALIBRATION
from commitjl.cost_aware import Planner
from validation.verify_cost_aware import validate_pins,validate_routes
from experiments.published_baseline_gate import DEFAULT,METHODS

REPO=Path(__file__).resolve().parents[2]


def prepare(root):
    out=root/'warm_followup_v2';out.mkdir(exist_ok=False)
    cases=[]
    for dataset in ['fashion','text']:
        for method in (METHODS if dataset=='fashion' else METHODS[::-1]):
            name=f'{dataset}_n32' if method=='ordered_head' else f'{dataset}_n32_{method}'
            cases.append(dict(name=name,dataset=dataset,method=method,requests=4,
                              run_full_baseline=method=='ordered_head'))
    save(out/'plan.json',dict(status='frozen_before_warm_followup_outcomes',cases=cases,
        source_gate_plan_sha256=sha(root/'plan.json'),
        registered_after_cold_native_outcomes=True,
        policy='Existing cost-aware policy,16 corrections and10% safety margin, identical old calibration for every basis.',
        calibration='Earlier seed521 Q8 pilot; no retuning to this new cohort.',
        timing='Fresh warm request proofs, key loads, native planning and public replay. Old evidence/setup reused equally. Copy/export and independent integer oracle outside phase sums.',
        baseline='One newly measured warm Full arm per dataset, shared as a comparator; no invented per-method repetitions.',
        repetitions=1,proof_budget_hours=2,serial=True,workers=8,
        implementation_sha256=sha(Path(__file__)),
        dependencies_sha256={p.relative_to(REPO).as_posix():sha(p) for p in [REPO/'code/campaigns/cost_aware_pilot.py',
            REPO/'code/commitjl/cost_aware.py',REPO/'code/validation/verify_cost_aware.py']}))
    save(out/'plan_pin.json',dict(plan_sha256=sha(out/'plan.json')))


class WarmCampaign(Campaign):
    def invoke(self,*args,**kwargs):
        if time.time()-self.state['started_unix']>2*3600:
            raise RuntimeError('Two-hour warm follow-up cap')
        return super().invoke(*args,**kwargs)


def run(root):
    assert os.name=='posix','Use the same Linux environment as the cold cohort'
    out=root/'warm_followup_v2';plan=read(out/'plan.json')
    assert sha(out/'plan.json')==read(out/'plan_pin.json')['plan_sha256']
    assert sha(Path(__file__))==plan['implementation_sha256']
    for name,digest in plan['dependencies_sha256'].items():assert sha(REPO/name)==digest
    if not read(root/'proofs/matched_queue.json').get('complete'):
        raise RuntimeError('Cold proof queue must complete before the serial warm follow-up')
    os.environ['COMMITJL_RESUME_PROOFS']='1'
    campaign=WarmCampaign(out)
    try:
        for case in plan['cases']:
            dataset,method,name=case['dataset'],case['method'],case['name']
            source=root/'proofs/matched'/f'{dataset}_n32'
            assert read(source/(method+'_refined/checked_comparison.json'))['complete']
            # A read-only alias adapts the older generic prover helper's path
            # names. All writes go to the new warm output, never these sources.
            alias=out/'sources'/name;alias.mkdir(parents=True)
            for role,target in [('ordered_head',source/method),('ordered_head_base',source/(method+'_base')),
                                ('ordered_head_refined',source/(method+'_refined'))]:
                (alias/role).symlink_to(target.resolve(),target_is_directory=True)
            target=out/name
            inp=read(source/(method+'_base/input.json'))
            task=read(source/(method+'_base/approved.json'))
            assert len(task['reuse_centers'])==case['requests']
            pins=dict(binary_sha256=sha(GO),certificate=read(source/(method+'_base/owner_pin.json')),
                      full=read(source/method/'full/owner_pin.json'))
            rpin=source/(method+'_refined/owner_refinement_pin.json')
            if rpin.exists():pins['refinement']=read(rpin)
            validate_pins(pins,task)
            costs,calibration=calibrate(CALIBRATION/'matched'/f'{dataset}_n32')
            planner=Planner(inp['model'],inp['x'],costs)
            routes=planner.batch(task['reuse_centers'])
            validate_routes({k:routes[k] for k in ['certificate','full']},len(task['reuse_centers']))
            save(target/'owner_pin.json',pins);save(target/'approved.json',task);save(target/'planner.json',routes)
            save(target/'calibration.json',dict(source=str(CALIBRATION/'matched'/f'{dataset}_n32'),
                 source_pins=calibration,costs=routes['costs']))
            prove_case(campaign,out,case,source=alias,run_baseline=case['run_full_baseline'])
        save(out/'checks.json',dict(complete=True,
            cases=[dict(dataset=c['dataset'],method=c['method'],name=c['name'],**read(out/c['name']/'checked.json')) for c in plan['cases']]))
        campaign.state.update(complete=True,status='complete',active_phase=None,finished_unix=time.time())
        save(out/'matched_queue.json',campaign.state)
    except BaseException as error:
        campaign.state.update(complete=False,status='stopped_with_preserved_evidence',error=str(error))
        save(out/'matched_queue.json',campaign.state)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','run'])
    parser.add_argument('--root',type=Path,default=DEFAULT);args=parser.parse_args()
    (prepare if args.phase=='prepare' else run)(args.root.resolve())
