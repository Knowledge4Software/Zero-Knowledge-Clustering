"""Serial follow-up queue for the user's explicit four-hour research window."""
from pathlib import Path
import argparse
import copy
import datetime
import json
import os
import shutil
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'code'))
from campaigns.published_baselines_laptop import Campaign,GO,RUST,read,save,sha
from campaigns.laptop_advantage_pilot import DEFAULT as BASIC,PRIOR

ROOT=REPO/'code/validation/outputs/laptop_four_hour_research_20261028'
END='2026-10-28T09:54:59+00:00'


def initialize(root):
    root.mkdir(parents=True,exist_ok=False)
    save(root/'plan.json',dict(
        user_request='Run local RQ-like experiments during the next four hours; determine whether each baseline can be beaten in any explicitly stated case.',
        observed_user_window_start_utc='2026-10-28T05:54:59+00:00',deadline_utc=END,
        timezone='Asia/Tehran',deadline_local='2026-10-28T13:24:59+03:30',
        serial=True,workers=8,rss_cap_mib=3584,minimum_available_ram_mib=650,
        existing_basic_campaign=str(BASIC.relative_to(REPO)),
        queue=[
            'Finish the existing2-dataset N32 eight-request matched real-proof pilot.',
            'Numerical rank/correction ablation on the same real sources; explicitly native-only.',
            'Fashion N128, first4 fixed proposals: all4 systems, fresh real proofs. Exploratory scale case selected after native feasibility/coverage screen.',
            'If at least90 minutes remain, Fashion N32 all64 precomputed candidates: actual certificate vs Halo2 warm comparison, all setup reuse disclosed and pinned.',
        ],
        selection='The scaled Fashion case is deliberately exploratory. Native controls suggested favorable geometry. Retain every outcome and failure; no claim of independent confirmation.',
        secondary_baselines='ezDPS and Zhang2021 are still not implemented as same-task systems; do not invent runtime claims.',
        reporting='For each implemented baseline give measured wins, ties, losses, infeasible or unmeasured, with exact metric/regime. Different backends/commitments and approximate versus exact outputs disclosed.',
        stop_directories=[str(BASIC.relative_to(REPO)),str((root/'scaled').relative_to(REPO)),
                          str((root/'warm64').relative_to(REPO)),str(root.relative_to(REPO))],
        no_remote_access=True,no_paper_edits=True,
    ))
    save(root/'plan_pin.json',dict(plan_sha256=sha(root/'plan.json'),script_sha256=sha(Path(__file__)),
         basic_plan_sha256=sha(BASIC/'plan.json'),go_sha256=sha(GO),rust_sha256=sha(RUST)))
    save(root/'queue.json',dict(status='waiting_for_basic_pilot',complete=False,created_unix=time.time(),active_phase=None))


class DeadlineCampaign(Campaign):
    def __init__(self,root,deadline):
        super().__init__(root);self.deadline=deadline
    def invoke(self,label,command,timeout=1800,expect_failure=False):
        remaining=self.deadline-time.time()
        if remaining<20:raise RuntimeError('Four-hour deadline; no new phase admitted')
        return super().invoke(label,command,min(timeout,remaining-5),expect_failure)


def prepare_scaled(root):
    import numpy as np
    target=root/'matched/fashion_n128'
    original=BASIC/'matched/fashion_n32'
    for method in ['ordered_head','pca']:
        archive=np.load(original/f'{method}_diagnostic_inputs.npz')
        inp=read(original/method/'head_r63/input.json')
        inp['model']['n']=128;inp['x']=archive['audit'][:128].tolist()
        inp['reuse_centers']=inp['reuse_centers'][:5]
        inp['queries']=[list(range(i,i+16)) for i in range(0,128,16)]
        save(target/method/'head_r63/input.json',inp)
        full=copy.deepcopy(inp);full['model'].update(method='full',basis=[],seed='')
        full['queries']=[list(range(i,i+8)) for i in range(0,128,8)]
        save(target/method/'full/input.json',full)
        protocol=read(original/method/'protocol.json')
        protocol.update(n=128,native_n=128,fitting_seconds=0.,phase_wall_limit_seconds=1800,
            selected_source_sets=list(range(4)),source_proposals=protocol['source_proposals'][:4],
            source_selection='Exploratory scale test: fixed first4 proposals, no selection among their audit outcomes.',
            fit_scope='Same frozen learned representations supplied to all methods; source fitting separately recorded, excluded equally from this service-scale comparison.')
        save(target/method/'protocol.json',protocol)
        aux=read(original/method/'auxiliary_basis.json')
        aux.update(source_input_sha256=sha(target/method/'head_r63/input.json'),fitting_seconds=0.)
        save(target/method/'auxiliary_basis.json',aux)
    inp=read(target/'ordered_head/full/input.json')
    assert inp==read(target/'pca/full/input.json')
    save(target/'approved_task.json',dict(reuse_centers=inp['reuse_centers']))
    save(target/'input_pin.json',dict(complete=True,n=128,d=784,k=5,
        full_input_sha256=sha(target/'ordered_head/full/input.json'),
        centroid_input_sha256=sha(target/'ordered_head/head_r63/input.json'),
        pca_input_sha256=sha(target/'pca/head_r63/input.json'),centers=5))
    save(root/'authorization_and_scope.json',read(BASIC/'authorization_and_scope.json'))
    for name in ['compact_smoke','refinement_smoke']:
        save(root/name/'checks.json',read(PRIOR/name/'checks.json'))
    return target


def ablations(root):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from commitjl.group_change_bounds import exact_old_correction_bounds
    from commitjl.group_moment_reference import prepare_groups
    from commitjl.jl_certificate_pilots import hint_order,refinement_work
    from experiments.pilot_jl_certificate_ideas import distances
    records=[]
    with threadpool_limits(limits=1):
        for name in ['fashion','text']:
            source=BASIC/'matched'/f'{name}_n32'
            for method in ['ordered_head','pca']:
                data=np.load(source/f'{method}_diagnostic_inputs.npz')
                base=read(source/method/'head_r63/input.json')['model'];old=data['centers'][0]
                for n in [32,128]:
                    x=data['audit'][:n];od=distances(x,old);groups=od.argmin(1)
                    prepared=prepare_groups(x,groups.tolist(),5);old_cost=int(od.min(1).sum())
                    for rank in [4,39,63,79]:
                        model=dict(base,n=n,basis=data['auxiliary_basis'][:rank].tolist())
                        for i,new in enumerate(data['centers'][1:9]):
                            nd=distances(x,new);truth=int(nd.min(1).sum())-old_cost
                            bounds=exact_old_correction_bounds(model,x,prepared,old,new);den=bounds['cost_denominator']
                            correction=[int(v)*den for v in nd.min(1)-nd[np.arange(n),groups]]
                            order=hint_order(bounds,None,'width')
                            for cap in [0,4,16]:
                                out=refinement_work(bounds,correction,order,cap)
                                assert out['lower']<=truth*den<=out['upper']
                                expected='improves' if truth<0 else 'worsens' if truth>0 else 'equal'
                                assert out['decision'] in ['inconclusive',expected]
                                records.append(dict(dataset=name,method=method,n=n,certificate_rank=rank,
                                    cap=cap,request=i,truth=truth,decision=out['decision'],exact_rows=out['exact_rows']))
    save(root/'native_ablations.json',dict(complete=True,native_only=True,new_proofs=0,
         frozen_prediction_rank=63,description='Certificate basis rank changes only. Prediction remains the frozen rank63 model; every exact cost and interval checked.',
         records=records,all_intervals_contain_truth=True,no_wrong_decisions=True))


def run(root):
    plan=read(root/'plan.json');pin=read(root/'plan_pin.json')
    assert sha(root/'plan.json')==pin['plan_sha256']
    deadline=datetime.datetime.fromisoformat(plan['deadline_utc']).timestamp()
    state=read(root/'queue.json')
    def status(phase,**kwargs):
        state.update(active_phase=phase,updated_unix=time.time(),**kwargs);save(root/'queue.json',state)
    try:
        while not read(BASIC/'matched_queue.json').get('complete'):
            basic=read(BASIC/'matched_queue.json')
            if basic.get('status')=='stopped_with_evidence':raise RuntimeError('Basic pilot stopped: '+basic.get('error',''))
            if time.time()>deadline-30:raise RuntimeError('Deadline reached while waiting for basic pilot')
            time.sleep(5)
        status('audit_basic',status='running')
        from analysis.laptop_advantage_pilot_report import main as basic_report
        basic_report(BASIC)
        status('native_ablations');ablations(root)
        remaining=deadline-time.time()
        if remaining>=90*60:
            status('warm64_ready',remaining_seconds=remaining)
            from campaigns.laptop_warm64_control import run as warm_run
            warm_run(root/'warm64',deadline)
        else:status('warm64_not_admitted',remaining_seconds=remaining)
        if deadline-time.time()<30*60:
            status('scaled_not_admitted',status='complete',complete=True,finished_unix=time.time())
            return
        status('scaled_fashion_n128')
        scaled=root/'scaled';target=prepare_scaled(scaled)
        campaign=DeadlineCampaign(scaled,deadline)
        try:
            campaign.rust_case(target);campaign.method_case(target)
            campaign.state.update(complete=True,status='complete',active_phase=None,finished_unix=time.time())
            save(scaled/'matched_queue.json',campaign.state)
        except Exception as error:
            campaign.state.update(complete=False,status='stopped_with_evidence',error=str(error))
            save(scaled/'matched_queue.json',campaign.state)
            status('scaled_failed',scaled_error=str(error))
        status('ready_for_final_report',status='complete',complete=True,finished_unix=time.time())
    except BaseException as error:
        status('stopped_with_evidence',status='stopped_with_evidence',complete=False,error=str(error))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--initialize',action='store_true')
    args=parser.parse_args()
    if args.initialize:initialize(args.root.resolve())
    else:run(args.root.resolve())
