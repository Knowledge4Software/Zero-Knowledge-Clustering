"""Run the cost-aware control on an approved task and existing authenticated case.

The case must have the completed compact/refinement and matched Full setup from
the earlier pilot. Their keys, rows and old-reference evidence are reused.
Every changing-request proof is generated anew; no baseline timing is invented.
"""
from pathlib import Path
import argparse
import datetime
import os
import time

from campaigns.published_baselines_laptop import GO, REPO, read, save, sha
from campaigns.cost_aware_pilot import calibrate, prove_case
from campaigns.laptop_four_hour_research import DeadlineCampaign
from commitjl.cost_aware import Planner
from validation.verify_cost_aware import validate_pins, validate_routes


def initialize(root, source, task_path):
    root.mkdir(parents=True, exist_ok=False)
    task = read(task_path)
    inp = read(source/'ordered_head_base/input.json')
    pin = dict(binary_sha256=sha(GO), certificate=read(source/'ordered_head_base/owner_pin.json'),
               full=read(source/'ordered_head/full/owner_pin.json'),
               refinement=read(source/'ordered_head_refined/owner_refinement_pin.json'))
    validate_pins(pin,task)
    costs, calibration = calibrate(source)
    planner = Planner(inp['model'],inp['x'],costs)
    plan = planner.batch(task['reuse_centers'])
    validate_routes({k:plan[k] for k in ['certificate','full']},len(task['reuse_centers']))
    out = root/source.name
    save(out/'owner_pin.json',pin);save(out/'approved.json',task);save(out/'planner.json',plan)
    save(out/'calibration.json',dict(source=str(source),sources_sha256=calibration,costs=plan['costs']))
    now=datetime.datetime.now(datetime.timezone.utc)
    deadline=now+datetime.timedelta(minutes=35)
    save(root/'plan.json',dict(created_utc=now.isoformat(),deadline_utc=deadline.isoformat(),
         source=str(source),name=source.name,requests=len(task['reuse_centers']),
         stop_directories=[str(root.relative_to(REPO))],workers=8,rss_cap_mib=3584,
         serial=True,no_paper_edits=True,no_remote_access=True,
         policy='Per-request conclusive/cost preflight followed by a batch activation check. Charge certificate/refinement key-loading overhead and require10% batch saving.',
         timing='Fresh warm routed proof generation, native planner, and public replay. No new baseline arm in this command.',
         limitation='Engineering follow-up to the first cost-aware pilot. Same exploratory data, no statistical generalization or universal speed claim.'))
    papers=[REPO/'manuscript.tex',REPO/'references.bib']+sorted((REPO/'paper_results').rglob('*.tex'))
    save(root/'preservation.json',{str(p.relative_to(REPO)):sha(p) for p in papers})
    files=[Path(__file__),REPO/'code/campaigns/cost_aware_pilot.py',REPO/'code/commitjl/cost_aware.py',REPO/'code/validation/verify_cost_aware.py']
    files+=list(root.rglob('*.json'))
    save(root/'source_pin.json',{str(p.relative_to(REPO)):sha(p) for p in files})
    print(dict(initialized=True,root=str(root),certificate=plan['certificate'],full=plan['full'],deadline_utc=deadline.isoformat()),flush=True)


def run(root):
    for p,digest in read(root/'source_pin.json').items():
        if sha(REPO/p)!=digest:raise RuntimeError('frozen input/source changed: '+p)
    plan=read(root/'plan.json')
    os.environ['COMMITJL_RESUME_PROOFS']='1'
    campaign=DeadlineCampaign(root,datetime.datetime.fromisoformat(plan['deadline_utc']).timestamp())
    try:
        prove_case(campaign,root,dict(name=plan['name'],requests=plan['requests']),
                   source=Path(plan['source']),run_baseline=False)
        assert all(sha(REPO/p)==digest for p,digest in read(root/'preservation.json').items())
        campaign.state.update(complete=True,status='complete',active_phase=None,finished_unix=time.time())
        save(root/'matched_queue.json',campaign.state)
        save(root/'checks.json',dict(complete=True,paper_unchanged=True,no_new_baseline_timing=True,
             result=read(root/plan['name']/'checked.json')))
    finally:(root/'FINISHED').write_text('Cost-aware warm control ended; preserve every receipt.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--task',type=Path)
    p.add_argument('--initialize',action='store_true');a=p.parse_args()
    if a.initialize:
        if a.source is None or a.task is None:p.error('initialization needs --source and --task')
        initialize(a.root.resolve(),a.source.resolve(),a.task.resolve())
    else:run(a.root.resolve())
