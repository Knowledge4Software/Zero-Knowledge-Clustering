"""Bounded16-request extension of the unfavorable text pilot.

Fresh real cold proofs for all4 systems; source code/data stay frozen.
Historical8-request evidence is used only for a labeled cost decomposition.
"""
from pathlib import Path
import argparse,copy,datetime,json,shutil,sys,time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'code'))
from campaigns.published_baselines_laptop import GO,RUST,read,save,sha
from campaigns.laptop_advantage_pilot import DEFAULT as BASIC,PRIOR
from campaigns.laptop_four_hour_research import DeadlineCampaign

ROOT=REPO/'code/validation/outputs/request_crossover_followup_20261028'


def initialize(root):
    import numpy as np
    root.mkdir(parents=True,exist_ok=False)
    now=datetime.datetime.now(datetime.timezone.utc);deadline=now+datetime.timedelta(minutes=90)
    save(root/'plan.json',dict(created_utc=now.isoformat(),deadline_utc=deadline.isoformat(),
        user_request='More requests on the slower text case; separate preprocessing from answering and estimate crossover honestly.',
        dataset='MiniLM text',n=32,requests=16,fit_seed=521,workers=8,serial=True,
        rss_cap_mib=3584,min_available_ram_mib=650,
        selection='Fixed first16 distinct chronological proposals from the previously frozen64; no favorable-outcome filtering.',
        prediction_rank=63,certificate_rank=79,max_corrections=16,
        methods=['exact Full Groth16','centroid certificate','PCA certificate','ZKML primitive adapter'],
        timing='Fresh cold generation/public replay; separately report initial authentication/keys, query work, refinement and paid Full fallback. Frozen fitted representations supplied to all methods.',
        limitations='One exploratory execution; previous8-request times and new16-request times are different executions, not paired statistical replications. Conditional cost models are not observed response times at untested counts.',
        stop_directories=[str(root.relative_to(REPO))],no_paper_edits=True,no_remote_access=True))
    source=BASIC/'matched/text_n32';target=root/'matched/text_n32'
    for method in ['ordered_head','pca']:
        archive=np.load(source/f'{method}_diagnostic_inputs.npz')
        inp=read(source/method/'head_r63/input.json');inp['reuse_centers']=archive['centers'][:17].tolist()
        assert len({json.dumps(c) for c in inp['reuse_centers']})==17
        save(target/method/'head_r63/input.json',inp)
        full=copy.deepcopy(inp);full['model'].update(method='full',basis=[],seed='')
        full['queries']=[list(range(i,i+8)) for i in range(0,32,8)]
        save(target/method/'full/input.json',full)
        protocol=read(source/method/'protocol.json');protocol.update(fitting_seconds=0.,
            selected_source_sets=list(range(16)),source_selection='Fixed first16 genuine proposals; all retained.',
            source_proposals=[dict(family='new_training_minibatch',batch=i//4,rate=[.125,.25,.5,1.][i%4]) for i in range(16)],
            fit_scope='Frozen fitted representations provided equally; no refitting during proof timing.')
        save(target/method/'protocol.json',protocol)
        aux=read(source/method/'auxiliary_basis.json');aux.update(fitting_seconds=0.,
            source_input_sha256=sha(target/method/'head_r63/input.json'))
        save(target/method/'auxiliary_basis.json',aux)
    inp=read(target/'ordered_head/full/input.json');assert inp==read(target/'pca/full/input.json')
    save(target/'approved_task.json',dict(reuse_centers=inp['reuse_centers']))
    save(target/'input_pin.json',dict(complete=True,n=32,d=384,k=5,centers=17,
        full_input_sha256=sha(target/'ordered_head/full/input.json'),
        centroid_input_sha256=sha(target/'ordered_head/head_r63/input.json'),
        pca_input_sha256=sha(target/'pca/head_r63/input.json')))
    papers=[REPO/'manuscript.tex',REPO/'references.bib']+sorted((REPO/'paper_results').rglob('*.tex'))
    save(root/'authorization_and_scope.json',dict(authorization='Current explicit request to extend requests and split timing.',
        paper_sha256={p.relative_to(REPO).as_posix():sha(p) for p in papers}))
    save(root/'pins.json',dict(plan_sha256=sha(root/'plan.json'),go_sha256=sha(GO),rust_sha256=sha(RUST),
        runner_sha256=sha(Path(__file__))))
    for name in ['compact_smoke','refinement_smoke']:
        check=read(PRIOR/name/'checks.json');assert check['complete'] and check['binary_sha256']==sha(GO)
        save(root/name/'checks.json',check)
    save(root/'queue.json',dict(complete=False,status='initialized'))


def run(root):
    plan=read(root/'plan.json');assert sha(root/'plan.json')==read(root/'pins.json')['plan_sha256']
    deadline=datetime.datetime.fromisoformat(plan['deadline_utc']).timestamp()
    campaign=DeadlineCampaign(root,deadline);target=root/'matched/text_n32'
    try:
        save(root/'queue.json',dict(complete=False,status='running',started_unix=time.time()))
        campaign.rust_case(target);campaign.method_case(target)
        campaign.state.update(complete=True,status='complete',active_phase=None,finished_unix=time.time())
        save(root/'matched_queue.json',campaign.state)
        from analysis.published_baseline_report import report
        report(root,output_stem='request_crossover_followup_20261028',emit=False,expected_case_count=1,
            limitations=[plan['limitations'],'ZKML uses the primitive adapter, not the unmodified compiler. Different backend/commitment and exact-cost versus sign outputs remain explicit.'])
        save(root/'queue.json',dict(complete=True,status='complete',finished_unix=time.time()))
    except BaseException as error:
        campaign.state.update(complete=False,status='stopped_with_evidence',error=str(error))
        save(root/'matched_queue.json',campaign.state)
        save(root/'queue.json',dict(complete=False,status='stopped_with_evidence',error=str(error),finished_unix=time.time()))
        raise
    finally:
        (root/'FINISHED').write_text('Bounded proof extension stopped; preserve all receipts.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--initialize',action='store_true');args=p.parse_args()
    if args.initialize:initialize(args.root.resolve())
    else:run(args.root.resolve())
