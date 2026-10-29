"""Separate measured reusable stages from responding; label extrapolation.

This is cost accounting from real proof receipts, not an additional benchmark.
It never reports a modeled threshold as a measured request-prefix crossover.
"""
from pathlib import Path
import argparse,json,math,sys,time

REPO=Path(__file__).resolve().parents[2]
OLD=REPO/'code/validation/outputs/laptop_advantage_pilot_20261028'
NEW=REPO/'code/validation/outputs/request_crossover_followup_20261028'
WARM=REPO/'code/validation/outputs/laptop_four_hour_research_20261028/warm64'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sample_work(s):
    return s.get('native_answer_seconds',0)+s.get('witness_seconds',0)+sum(s.get('prove_seconds',[]))+sum(s.get('verify_seconds',[]))
def resource(root,label):
    values=[read(p) for p in sorted((root/'resources').glob(label+'.attempt_*.json'))]
    assert values and values[-1]['complete'] and all(v['stop'] is None for v in values),label
    return sum(v['wall_seconds'] for v in values)


def extract(root,name):
    p=root/'matched'/name
    rows=[]
    full_file=p/'ordered_head/full/bundle/full_reuse_8.json'
    if full_file.exists() and read(full_file).get('complete'):
        f=read(full_file);count=len(read(p/'ordered_head/full/input.json')['reuse_centers'])-1
        samples=f['samples'];old=sum(sample_work(s) for s in samples if s.get('centre_set',0)==0)
        prep=read(p/'ordered_head/full/prepare.resource.json')['wall_seconds']
        initial=prep+f['compile_seconds']+f['setup_seconds']+old
        phases=sum(read(p/'ordered_head/full'/(phase+'.resource.json'))['wall_seconds'] for phase in ['prepare','full_reuse','verify'])
        measured=resource(root,name+'_groth_full')
        rows.append(dict(case=name,requests=count,method='Full-Groth16',
            native_summary_seconds=0.,summary_authentication_seconds=0.,
            native_owner_prepare_seconds=prep,base_key_compile_setup_seconds=f['compile_seconds']+f['setup_seconds'],
            refinement_key_seconds=0.,fallback_setup_seconds=0.,old_reference_proof_seconds=old,
            identified_reusable_seconds=initial,response_residual_seconds=phases-initial,
            response_residual_mean_seconds=(phases-initial)/count,
            directly_timed_recurring_calls_mean_seconds=sum(sample_work(s) for s in samples if s.get('centre_set',0)>0)/count,
            phase_total_seconds=phases,enclosing_invocation_seconds=measured,
            unclassified_enclosing_seconds=measured-phases,fallback_updates=0,
            native_summary_scope='Full does not construct the proposed compressed summary.'))
    if (p/'zkml_checked.json').exists():
        q=read(p/'zkml_checked.json');count=q['proofs']-1;setup=read(p/'zkml_private/setup.json')
        prep=resource(root,name+'_zkml_prepare')-setup['mock_check_seconds']
        old=read(p/'zkml_private/bank_000/result.json')['total_seconds']
        oldverify=resource(root,name+'_zkml_verify000')
        total=prep+resource(root,name+'_zkml_queries')+sum(resource(root,name+f'_zkml_verify{i:03}') for i in range(count+1))
        initial=prep+old+oldverify
        rows.append(dict(case=name,requests=count,method='ZKML-adapter',
            native_summary_seconds=0.,summary_authentication_seconds=0.,
            native_owner_prepare_seconds=setup['owner_preparation_seconds'],
            base_key_compile_setup_seconds=prep-setup['owner_preparation_seconds'],
            refinement_key_seconds=0.,fallback_setup_seconds=0.,old_reference_proof_seconds=old+oldverify,
            identified_reusable_seconds=initial,response_residual_seconds=total-initial,
            response_residual_mean_seconds=(total-initial)/count,
            directly_timed_recurring_calls_mean_seconds=sum(read(p/f'zkml_private/bank_{i:03}/result.json')['total_seconds'] for i in range(1,count+1))/count,
            phase_total_seconds=total,enclosing_invocation_seconds=total,
            unclassified_enclosing_seconds=0.,fallback_updates=0,
            native_summary_scope='Catalog commitment and exact inference circuit, not the proposed compressed summary.'))
    for method in ['ordered_head','pca']:
        checked_path=p/(method+'_refined/checked_comparison.json')
        if not checked_path.exists():continue
        checked=read(checked_path);assert checked['complete']
        base=read(p/(method+'_base/status.json'))['resources']
        rs=read(p/(method+'_refined/status.json'))['resources']
        native=base['group_compact_prepare']['wall_seconds'];keys=base['group_keys']['wall_seconds']
        auth=base['group_rows']['wall_seconds'];refkey=rs.get('refinement_setup',{}).get('wall_seconds',0.)
        fallback_initial=0.;fallback_calls=0.;fallback_old=0.
        fb=p/(method+'_refined/fallback/full')
        if (fb/'bundle/full_reuse_8.json').exists():
            f=read(fb/'bundle/full_reuse_8.json')
            fallback_old=sum(sample_work(s) for s in f['samples'] if s.get('centre_set',0)==0)
            fallback_initial=read(fb/'prepare.resource.json')['wall_seconds']+f['compile_seconds']+f['setup_seconds']+fallback_old
            fallback_calls=sum(sample_work(s) for s in f['samples'] if s.get('centre_set',0)>0)
        fitting=read(p/(method+'_base/protocol.json'))['fitting_seconds']
        initial=native+keys+auth+refkey+fallback_initial+fitting
        count=checked['proposal_updates'];phases=checked['active_phase_seconds_including_fitting_and_fallback']
        measured=resource(root,name+'_'+method+'_base')+resource(root,name+'_'+method+'_refined')
        recurring=sum(sample_work(s) for s in read(p/(method+'_base/bundle/group_query.json'))['samples'])+fallback_calls
        if (p/(method+'_refined/bundle/refinement_prove.json')).exists():
            recurring+=sum(sample_work(s) for s in read(p/(method+'_refined/bundle/refinement_prove.json'))['samples'])
        rows.append(dict(case=name,requests=count,method='Centroid' if method=='ordered_head' else 'PCA-certificate',
            native_summary_seconds=native,summary_authentication_seconds=auth,
            native_owner_prepare_seconds=native,base_key_compile_setup_seconds=keys,
            refinement_key_seconds=refkey,fallback_setup_seconds=fallback_initial-fallback_old,
            old_reference_proof_seconds=fallback_old,fitting_seconds=fitting,
            identified_reusable_seconds=initial,response_residual_seconds=phases-initial,
            response_residual_mean_seconds=(phases-initial)/count,
            directly_timed_recurring_calls_mean_seconds=recurring/count,
            phase_total_seconds=phases,enclosing_invocation_seconds=measured,
            unclassified_enclosing_seconds=measured-phases,fallback_updates=checked['fallback_updates'],
            refinement_proofs=checked['new_refinement_proofs'],
            native_summary_scope='Summary construction/commitment and its row-authentication proofs are separate from key generation.'))
    for row in rows:
        assert row['identified_reusable_seconds']>0 and row['response_residual_seconds']>=0
        assert abs(row['identified_reusable_seconds']+row['response_residual_seconds']-row['phase_total_seconds'])<.001
    return rows


def compare(group):
    ours=next((r for r in group if r['method']=='Centroid'),None)
    if ours is None:return []
    results=[]
    for other in group:
        if other is ours:continue
        delta=other['response_residual_mean_seconds']-ours['response_residual_mean_seconds']
        if delta>0:
            qstar=max(1,math.floor((ours['identified_reusable_seconds']-other['identified_reusable_seconds'])/delta)+1)
            conclusion='Estimated eventual crossover under unchanged average workload; not a measured threshold.'
        else:qstar=None;conclusion='No eventual crossover in this constant-mean model. More requests alone do not change the slope; future workload mixes can differ.'
        results.append(dict(case=ours['case'],calibration_requests=ours['requests'],baseline=other['method'],
            estimated_first_better_request=qstar,conclusion=conclusion,
            small_margin=(abs(delta)<.05*max(ours['response_residual_mean_seconds'],other['response_residual_mean_seconds'])),
            our_mean_response_seconds=ours['response_residual_mean_seconds'],
            baseline_mean_response_seconds=other['response_residual_mean_seconds']))
    return results


def build():
    records=[]
    for name in ['fashion_n32','text_n32']:
        for row in extract(OLD,name):records.append(dict(row,execution='previous8'))
    if (NEW/'matched/text_n32').exists():
        for row in extract(NEW,'text_n32'):records.append(dict(row,execution='new16'))
    crossovers=[]
    for execution,case in sorted({(r['execution'],r['case']) for r in records}):
        crossovers+=compare([r for r in records if r['execution']==execution and r['case']==case])
    warm=read(WARM/'checked_comparison.json')
    image_ours=next(r for r in records if r['case']=='fashion_n32' and r['method']=='Centroid')
    image_exact=next(r for r in records if r['case']=='fashion_n32' and r['method']=='ZKML-adapter')
    cold64=dict(our_seconds=image_ours['identified_reusable_seconds']+warm['measured_warm_certificate_seconds'],
                zkml_seconds=image_exact['identified_reusable_seconds']+warm['measured_warm_exact_seconds'],
                qualification='Component reconstruction from measured setup and separately measured warm64 work using the same cached epoch; not a new cold64 run or an exact crossover measurement.')
    result=dict(records=records,conditional_crossover_estimates=crossovers,
        new16_status=read(NEW/'queue.json') if (NEW/'queue.json').exists() else None,
        timing_definition='Identified reusable stages versus residual response-phase work. Key serialization/loading, file I/O and public replay not independently isolated remain in the response residual. This is component accounting, not independently timed single-request latency.',
        model='T(Q)=P+Q*c using the observed mixture and batching policy. Crossover counts are conditional estimates, not measured guarantees. Fallback and correction costs are retained.',
        reuse_scope='Each request evaluates new public centers against one fixed catalog and old reference. Changing the catalog/reference requires charging new authentication in the current implementation.',
        measured_warm64_fashion=dict(our_seconds=warm['measured_warm_certificate_seconds'],
            zkml_seconds=warm['measured_warm_exact_seconds'],full_groth16_measured=False),
        reconstructed_cold64_fashion=cold64,
        frozen_fit_note='The fresh16-request study supplies the previously fitted representations to all systems. Summary construction/authentication and proof keys are newly measured; basis-training time is not silently labeled compression.',
        paper_unchanged=True)
    target=REPO/'output/request_cost_split_20261028.json';target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines=['# Preparation versus request work — 28 October 2026','',result['timing_definition'],'',result['reuse_scope'],'',
        'All times below are seconds. Identified reusable work includes native commitment preparation, base keys, authentication of old rows, lazy correction-key setup and any measured exact-fallback key/old-reference proof. The response column retains all residual query, correction, fallback and verification work.','',
        '| Execution | Data | Requests | Method | Native data/summary preparation | Summary-authentication proofs | Base keys | Other reusable stages | Identified reusable total | Mean residual response | Enclosing accounting gap | Measured enclosing total |',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        parts=r['native_owner_prepare_seconds']+r['summary_authentication_seconds']+r['base_key_compile_setup_seconds']
        other=r['identified_reusable_seconds']-parts
        lines.append(f"| {r['execution']} | {r['case']} | {r['requests']} | {r['method']} | {r['native_owner_prepare_seconds']:.2f} | {r['summary_authentication_seconds']:.2f} | {r['base_key_compile_setup_seconds']:.2f} | {other:.2f} | {r['identified_reusable_seconds']:.2f} | {r['response_residual_mean_seconds']:.2f} | {r['unclassified_enclosing_seconds']:.2f} | {r['enclosing_invocation_seconds']:.2f} |")
    lines+=['','## Conditional crossover model','',result['model'],'',
            '| Data / calibration | Baseline | Estimated first better count | Interpretation |',
            '|---|---|---:|---|']
    for r in crossovers:
        point=('Not reliable: below5% single-run margin' if r['small_margin'] else
               f"Already favorable at measured Q={r['calibration_requests']}; lower-count crossing unmeasured" if r['estimated_first_better_request'] is not None and r['estimated_first_better_request']<r['calibration_requests'] else
               str(r['estimated_first_better_request']) if r['estimated_first_better_request'] is not None else 'No eventual crossing in model')
        lines.append(f"| {r['case']} / Q{r['calibration_requests']} | {r['baseline']} | {point} | {r['conclusion']} |")
    lines+=['','These count estimates must not be presented as measured crossover thresholds. The first8 and first16 prefixes can have different proposal mixtures. Full and ZKML also reuse their setup in the model.','',
            'Differences below5% in a single execution, particularly the PCA comparison, do not establish a reliable speed advantage or crossover count.','',
            f"The independently measured Fashion64 warm comparison remains {warm['measured_warm_certificate_seconds']/60:.2f}min for the certificate versus {warm['measured_warm_exact_seconds']/60:.2f}min for the ZKML adapter. There is no measured Full-Groth16 Q64 run.",'',
            f"Adding the separately measured reusable setup reconstructs about {cold64['our_seconds']/60:.2f}min versus {cold64['zkml_seconds']/60:.2f}min at Fashion Q64. This is component accounting, not a fresh cold64 execution. It also shows why the Q8 constant-mean model must not be treated as an exact threshold prediction.",'',
            '## Actual larger text run','',json.dumps(result['new16_status'],ensure_ascii=False),'',
            result['frozen_fit_note'],'','The paper is unchanged. No cloud resources were used.']
    (REPO/'output/request_cost_split_20261028.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plot(records)
    print(json.dumps(dict(records=len(records),crossovers=crossovers,new16_status=result['new16_status']),indent=2))


def plot(records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'Centroid':'#1769aa','Full-Groth16':'#278447','ZKML-adapter':'#d47714','PCA-certificate':'#777777'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,case in zip(axes,['fashion_n32','text_n32']):
        candidates=[r for r in records if r['case']==case]
        calibration=max(r['requests'] for r in candidates)
        rows=[r for r in candidates if r['requests']==calibration]
        for r in rows:
            q=list(range(calibration,129));p=r['identified_reusable_seconds'];c=r['response_residual_mean_seconds']
            ax.plot(q,[(p+x*c)/60 for x in q],'--',color=colors[r['method']],label=r['method'])
            ax.scatter([r['requests']],[r['phase_total_seconds']/60],color=colors[r['method']],s=40,zorder=3)
        ax.set_title(f"{case}: calibrated at Q={calibration}")
        ax.set_xlabel('Number of distinct candidate requests (Q)');ax.set_ylabel('Phase-accounted cumulative minutes')
        ax.grid(alpha=.22);ax.legend(fontsize=8)
    fig.suptitle('Conditional cost model: measured reusable work + Q x observed mean response')
    fig.text(.5,.015,'Dashed = unmeasured extrapolation; dots = measured calibration totals. Workload and fallback mixture assumed unchanged.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.93))
    fig.savefig(REPO/'output/request_cost_split_20261028.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watch',action='store_true');args=p.parse_args()
    if args.watch:
        while True:
            state=read(NEW/'queue.json')
            if state.get('complete') or state.get('status')=='stopped_with_evidence':break
            time.sleep(5)
    build()
