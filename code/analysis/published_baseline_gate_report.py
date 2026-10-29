"""Completed-only report for the local published-baseline gate."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

from commitjl.pilot_io import save, sha
from experiments.published_baseline_gate import DEFAULT, METHODS, SOURCES

REPO=Path(__file__).resolve().parents[2]
LABELS={'ordered_head_certificate':'Current centroid certificate',
        'pca_certificate':'PCA certificate',
        'sign_rp_certificate':'Sign RP + our certificate adapter',
        'sparse_rp_certificate':'Sparse RP + our certificate adapter',
        'sign_rp_raw':'Boutsidis sign RP (raw)',
        'sparse_rp_raw':'Achlioptas sparse RP (raw)',
        'lightweight_coreset_m16':'Bachem lightweight coreset,16 draws',
        'lightweight_coreset_m32':'Bachem lightweight coreset,32 draws',
        'uniform_coreset_m16':'Uniform sample,16 draws',
        'uniform_coreset_m32':'Uniform sample,32 draws',
        'rabitq_raw':'RaBitQ one-bit estimator (2024)',
        'exact_full':'Exact Full (native)'}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def number(v,scale=1,decimals=2):
    return 'N/A' if v is None else f'{v*scale:.{decimals}f}'


def summarize_native(data):
    groups={}
    for r in data['records']:
        key=(r['dataset'],r['n'],r['method']);groups.setdefault(key,[]).append(r)
    summary=[];by_fit=[]
    for (dataset,n,method),rows in sorted(groups.items()):
        agreement=[r['assignment_agreement'] for r in rows if r['assignment_agreement'] is not None]
        label_coverage=[r.get('label_certificate_coverage') for r in rows if r.get('label_certificate_coverage') is not None]
        conclusive=[r['conclusive'] for r in rows if r['conclusive'] is not None]
        base=[r['base_conclusive'] for r in rows if r['base_conclusive'] is not None]
        timing=[r for r in data['timings'] if r['case'].startswith(dataset+'_') and r['n']==n and r['method']==method]
        phase_ms={phase:statistics.median(statistics.median(sum(t['seconds'],[]))*1000 for t in timing if t['phase']==phase)
                  for phase in sorted({t['phase'] for t in timing})}
        classes={name:[r for r in rows if ('improves' if r['true_change']<0 else 'worsens' if r['true_change']>0 else 'equal')==name]
                 for name in ['improves','worsens','equal']}
        balanced=statistics.mean(statistics.mean(r['point_sign_correct'] for r in group) for group in classes.values() if group)
        record=dict(dataset=dataset,n=n,method=method,requests=len(rows),fits=len({r['fit_seed'] for r in rows}),
            assignment_agreement=statistics.mean(agreement) if agreement else None,
            point_sign_agreement=statistics.mean(r['point_sign_correct'] for r in rows),
            balanced_sign_agreement=balanced,
            always_worsens_agreement=statistics.mean(r['true_change']>0 for r in rows),
            false_improvement_predictions=sum(r['estimated_change']<0 and r['true_change']>=0 for r in rows),
            missed_improvement_predictions=sum(r['estimated_change']>=0 and r['true_change']<0 for r in rows),
            true_sign_counts={name:len(group) for name,group in classes.items()},
            label_certificate_coverage=statistics.mean(label_coverage) if label_coverage else None,
            base_coverage=statistics.mean(base) if base else None,
            refined_coverage=statistics.mean(conclusive) if conclusive else None,
            corrections=sum(r['corrections'] or 0 for r in rows) if conclusive else None,
            median_native_query_ms=phase_ms,
            wrong_estimated_signs=sum(not r['point_sign_correct'] for r in rows),
            statistical_coverage=(statistics.mean(r['statistical_conclusive'] for r in rows)
                                  if all('statistical_conclusive' in r for r in rows) else None),
            estimated_delta_absolute_error=statistics.mean(abs(r['estimated_change']-r['true_change']) for r in rows))
        summary.append(record)
        for seed in sorted({r['fit_seed'] for r in rows}):
            fit=[r for r in rows if r['fit_seed']==seed]
            agrees=[r['assignment_agreement'] for r in fit if r['assignment_agreement'] is not None]
            covered=[r['conclusive'] for r in fit if r['conclusive'] is not None]
            by_fit.append(dict(dataset=dataset,n=n,method=method,fit_seed=seed,requests=len(fit),
                assignment_agreement=statistics.mean(agrees) if agrees else None,
                point_sign_agreement=statistics.mean(r['point_sign_correct'] for r in fit),
                refined_coverage=statistics.mean(covered) if covered else None))
    return summary,by_fit


def proof_rows(root):
    """Use independently checked completed receipts, never partial predictions."""
    rows=[]
    proof=root/'proofs'
    def outer_seconds(label):
        receipts=sorted((proof/'resources').glob(label+'.attempt_*.json'))
        if not receipts or not all(read(p).get('complete') for p in receipts):return None
        return sum(read(p)['wall_seconds'] for p in receipts)
    for case in sorted((proof/'matched').glob('*_n*')):
        dataset=case.name.split('_n')[0]
        full=case/'ordered_head/full/public_bundle/verify.json'
        full_seconds=None
        if full.exists() and read(full).get('complete'):
            resources=[case/'ordered_head/full'/f'{phase}.resource.json' for phase in ['prepare','full_reuse','verify']]
            if all(p.exists() and read(p).get('complete') for p in resources):
                active_seconds=sum(read(p)['wall_seconds'] for p in resources)
                full_seconds=outer_seconds(dataset+'_full')
                inp=read(case/'ordered_head/full/input.json')
                assert full_seconds is not None
                rows.append(dict(dataset=dataset,method='Exact Full / Groth16',complete=True,active_phase_seconds=active_seconds,
                    requests=len(inp['reuse_centers'])-1,seconds=full_seconds,coverage=len(inp['reuse_centers'])-1,
                    fallback=0,correct_decisions=True,scope='Cold enclosing invocation wall; frozen model supplied; prepare/prove/public replay.'))
        if full_seconds is None:
            rows.append(dict(dataset=dataset,method='Exact Full / Groth16',complete=False))
        for method in METHODS:
            path=case/(method+'_refined/checked_comparison.json')
            if path.exists():
                c=read(path)
                assert c['complete'] and c['all_intervals_contain_full'] and c['no_wrong_decisions']
                base_wall=outer_seconds(dataset+'_'+method+'_base')
                refined_wall=outer_seconds(dataset+'_'+method+'_refined')
                assert base_wall is not None and refined_wall is not None
                fitting=read(case/method/'protocol.json')['fitting_seconds']
                seconds=base_wall+refined_wall+fitting
                rows.append(dict(dataset=dataset,method=LABELS[method+'_certificate'],complete=True,
                    active_phase_seconds=c['active_phase_seconds_including_fitting_and_fallback'],
                    requests=c['proposal_updates'],seconds=seconds,coverage=c['conclusive_without_fallback'],fallback=c['fallback_updates'],
                    corrections=c['new_refinement_proofs'],correct_decisions=True,full_over_method=full_seconds/seconds if full_seconds else None,
                    scope='Cold enclosing invocation walls plus basis fitting; correction, validation and freshly generated fallback included. Shared Full denominator is one completed run.'))
            else:
                rows.append(dict(dataset=dataset,method=LABELS[method+'_certificate'],complete=False))
        zk=case/'zkml_checked.json'
        if zk.exists() and read(zk).get('complete'):
            receipts=[]
            for label in [case.name+'_zkml_prepare',case.name+'_zkml_queries']:
                matches=sorted((proof/'resources').glob(label+'.attempt_*.json'))
                assert matches and all(read(p)['complete'] for p in matches)
                receipts.extend(matches)
            for label in sorted((proof/'resources').glob(case.name+'_zkml_verify*.attempt_*.json')):
                assert read(label)['complete'];receipts.append(label)
            setup=read(case/'zkml_private/setup.json');c=read(zk)
            seconds=sum(read(p)['wall_seconds'] for p in receipts)-setup['mock_check_seconds']
            rows.append(dict(dataset=dataset,method='ZKML primitive adapter / Halo2',complete=True,
                requests=len(c['costs'])-1,seconds=seconds,coverage=len(c['costs'])-1,fallback=0,
                correct_decisions=True,full_over_method=full_seconds/seconds if full_seconds else None,
                scope='Cold outer invocation totals; diagnostic MockProver excluded, parameter generation included. Different exact-cost output/backend/commitment.'))
        else:rows.append(dict(dataset=dataset,method='ZKML primitive adapter / Halo2',complete=False))
    return rows


def warm_rows(root):
    warm=root/'warm_followup_v2'
    if not (warm/'plan.json').exists():return [],False
    plan=read(warm/'plan.json')
    assert sha(warm/'plan.json')==read(warm/'plan_pin.json')['plan_sha256']
    results=[];full={}
    for case in plan['cases']:
        path=warm/case['name']/'checked.json'
        if not path.exists():continue
        result=read(path)
        assert result['complete'] and result['all_requests_verified'] and result['all_decisions_correct']
        results.append(dict(dataset=case['dataset'],method=case['method'],**result))
        if result['measured_warm_full_seconds'] is not None:
            full[case['dataset']]=result['measured_warm_full_seconds']
    for result in results:
        baseline=full.get(result['dataset'])
        result['shared_warm_full_seconds']=baseline
        result['shared_full_over_routed']=(baseline/result['measured_warm_routed_seconds'] if baseline is not None else None)
    queue=warm/'matched_queue.json'
    return results,queue.exists() and read(queue).get('complete',False)


def report(root,stem='published_baseline_gate_20261029'):
    plan=read(root/'plan.json');native=read(root/'native_results.json')
    assert native['complete'] and native['native_only'] and native['new_proofs']==0
    for case,digest in native['source_pins'].items():assert sha(root/'inputs'/case/'pin.json')==digest
    changed=[name for name,digest in plan['paper_sha256'].items() if sha(REPO/name)!=digest]
    assert not changed,changed
    combined=dict(native,records=list(native['records']),timings=list(native['timings']))
    rabitq_path=root/'rabitq_followup/results.json'
    rabitq=None
    if rabitq_path.exists():
        rabitq=read(rabitq_path);assert rabitq['complete'] and rabitq['native_only'] and rabitq['new_proofs']==0
        rp=read(root/'rabitq_followup/plan.json')
        assert sha(root/'rabitq_followup/plan.json')==read(root/'rabitq_followup/plan_pin.json')['plan_sha256']
        for case,digest in rp['input_pins'].items():assert sha(root/'inputs'/case/'pin.json')==digest
        for row in rabitq['records']:
            combined['records'].append(dict(row,method='rabitq_raw',deterministic_decision=None,
                base_conclusive=None,conclusive=None,corrections=None,proof_available=False))
        for row in rabitq['timings']:
            combined['timings'].append(dict(row,method='rabitq_raw',phase='query'))
    summary,by_fit=summarize_native(combined);proof=proof_rows(root);warm,warm_complete=warm_rows(root)
    queue_path=root/'proofs/matched_queue.json';queue=read(queue_path) if queue_path.exists() else {}
    test_receipt=root/'resources/tests.attempt_01.json'
    tests=read(test_receipt) if test_receipt.exists() else None
    test_log=root/'resources/tests.attempt_01.log'
    test_summary=(test_log.read_text(encoding='utf-8').strip().splitlines()[-1]
                  if test_log.exists() and test_log.stat().st_size else None)
    environment=read(root/'environment.json') if (root/'environment.json').exists() else None
    truth={}
    for r in native['records']:
        if r['method']!='exact_full':continue
        key=f'{r["dataset"]}_n{r["n"]}'
        counts=truth.setdefault(key,dict(improves=0,worsens=0,equal=0))
        counts['improves' if r['true_change']<0 else 'worsens' if r['true_change']>0 else 'equal']+=1
    value=dict(complete=bool(queue.get('complete',False) and rabitq is not None and warm_complete),
        report_generated=True,native_complete=True,proof_campaign_complete=queue.get('complete',False),
        native_summary=summary,native_per_fit=by_fit,proof_rows=proof,truth_classes=truth,
        rabitq_followup_complete=rabitq is not None,
        warm_comparison_complete=warm_complete,warm_rows=warm,
        validation_tests=tests,validation_test_summary=test_summary,measurement_environment=environment,
        paper_and_methodology_unchanged=True,papers=dict(SOURCES,zhang2021=dict(SOURCES['zhang2021'],
            implementation='Applicability review only: multiparty outsourced clustering using multiplication triples and garbled circuits, with a different statement and interaction contract.'),rabitq=dict(
            title='RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search',
            authors='Gao and Long',venue='SIGMOD2024',url='https://arxiv.org/abs/2405.12497')),
        scope='New exploratory fits, three per source. Native and proof times are separate. No unmodified external-system superiority claim.')
    save(REPO/'output'/f'{stem}.json',value)
    for suffix,records in [('native_per_fit',by_fit),('native_requests',combined['records'])]:
        keys=set().union(*(r.keys() for r in records))-{'interval'}
        # Convert set difference to a stable sequence for CSV export.
        keys=sorted(keys)
        with (REPO/'output'/f'{stem}_{suffix}.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    lines=['# Local gate against paper-derived baselines — 29 October 2026','',
        f'Study status: native complete; cold proof queue complete={queue.get("complete",False)}; RaBitQ follow-up complete={rabitq is not None}; equal-routing warm follow-up complete={warm_complete}.','',
        'The current four-stage methodology and all manuscript sources are unchanged. This gate uses new, independently refitted configurations of previously inspected public sources. It is exploratory evidence, not untouched confirmation.','',
        'Accuracy below means agreement with exact quantized nearest-center labels or the sign of exact quantized cost change. It is not ground-truth class accuracy. Coresets estimate aggregate costs and do not return labels for unsampled records.','',
        '## Assessment before renting a server','',
        'The current method has a useful verified-reuse operating region, but these results do not establish broad superiority over published systems. Its nearest-center label agreement exceeds the raw random sketches and one-bit RaBitQ on these fitted cohorts. Its deterministic decision coverage is strong for Fashion and HAR, while text is difficult. PCA is a near tie in the measured proof cohort. The measured ZKML primitive adapter is faster in both cold cohorts, with a different output/backend/commitment contract.','',
        'Small coresets can estimate aggregate decisions well without producing a deterministic, authenticated full-catalog sign certificate. Their speed and sign accuracy are useful controls, not interchangeable verified services. The text point predictor also needs caution: overall sign agreement can equal the majority-class baseline, as the balanced metric below shows.','',
        'A larger server campaign is justified to measure replication, scale and reuse economics if this narrower contribution is the intended claim. It is not justified by a demonstrated advantage of the current basis over PCA. The present proof cohort has only one fit and one timing execution per cell; none of the small differences should be read as statistically established.','',
        '## Added published algorithms','']
    for key in ['sign_rp','sparse_rp','lightweight_coreset']:
        s=SOURCES[key];lines.append(f'- **{s["title"]}** ({s["authors"]}, {s["venue"]}). [Source]({s["url"]}). {s["implementation"]}')
    lines.append('- **RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search** (Gao and Long,SIGMOD2024). [Paper](https://arxiv.org/abs/2405.12497), [author implementation](https://github.com/gaoj0017/RaBitQ). Separate numerical follow-up, registered after the original gate\'s native outcomes. Independent float64 one-bit estimator with old groups as IVF anchors; exhaustive all-center scan, no SIMD or query quantization. It is not the full ANN engine.')
    lines.append('- **ZKML: An Optimizing System for ML Inference in Zero-Knowledge Proofs** (EuroSys2024). [Paper](https://doi.org/10.1145/3627703.3650088), [author implementation](https://github.com/uiuc-kang-lab/zkml). Existing exact-cost primitive adapter is measured on the new matched cohort; this is not the complete published optimizing compiler.')
    lines+=['','Raw projections and coresets do not provide a deterministic full-catalog sign certificate at these empirical budgets. Their absence of a certificate is not a measured certificate failure. Random-projection certificate arms attach our existing authenticated relation and bounded refinement; they are adapters, not published ZK systems. Their residual-aware predictor is different from raw projected-distance prediction.','',
        '[Standalone scientific summary](./'+stem+'_summary.png) ([editable vector](./'+stem+'_summary.svg)). The chart separates label agreement, estimated decisions, deterministic coverage and cold proof-service time.','',
        '## Native quality and request cost','',
        'Each table pools eight candidates from each of three fits. Per-fit rows are supplied in the CSV. Median milliseconds summarize three timed passes within each fit, then the median across fits. Timing repetitions and correlated candidates are not independent learning replications. Cold basis fitting and reusable numerical preparation are retained in the raw JSON, outside these recurring query times.','']
    for dataset in plan['datasets']:
        lines += [f'### {dataset},128 records','',
            '| Method | Label agreement % | Estimated-sign agreement % | Balanced sign % | Base coverage % | After16 corrections % | Native cost/prediction ms | Native bound/refine ms |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in summary:
            if r['dataset']!=dataset or r['n']!=128:continue
            if r['method'] in ['sign_rp_certificate','sparse_rp_certificate']:continue
            timing=r['median_native_query_ms'];phase='prediction' if 'prediction' in timing else 'query'
            lines.append(f'| {LABELS[r["method"]]} | {number(r["assignment_agreement"],100)} | {number(r["point_sign_agreement"],100)} | {number(r["balanced_sign_agreement"],100)} | {number(r["base_coverage"],100)} | {number(r["refined_coverage"],100)} | {number(timing.get(phase),decimals=3)} | {number(timing.get("certificate"),decimals=3)} |')
        lines.append('')
        counts=truth[f'{dataset}_n128'];lines.append(f'True decision mix: {counts["improves"]} improvements, {counts["worsens"]} worsenings, {counts["equal"]} equalities. Always predicting worsening scores {100*counts["worsens"]/sum(counts.values()):.2f}%. Balanced sign agreement averages recall over the observed true classes; it is a post-hoc descriptive analysis of the frozen outcomes. False-improvement and missed-improvement counts are in JSON. Native certificate time is mathematical bound/refinement work, without cryptographic authentication or proving. It cannot be compared with ZK seconds.')
        lines.append('')
    lines += ['### Certificate adapters,128 records','',
        'The following basis controls share our deterministic interval relation. These are not raw published predictors or independent published proof systems. In particular, a nonorthogonal JL matrix can have useful raw distance estimates while its residual norm is too large for this particular deterministic envelope. Poor adapter coverage does not refute the original randomized approximation theorem.','',
        '| Data | Basis in our wrapper | Base coverage % | After16 corrections % | Native bound/refine ms |',
        '|---|---|---:|---:|---:|']
    for r in summary:
        if r['n']!=128 or not r['method'].endswith('_certificate'):continue
        lines.append(f'| {r["dataset"]} | {LABELS[r["method"]]} | {number(r["base_coverage"],100)} | {number(r["refined_coverage"],100)} | {number(r["median_native_query_ms"].get("certificate"),decimals=3)} |')
    lines.append('')
    lines += ['### RaBitQ probabilistic interval follow-up','',
        'RaBitQ uses its one-bit distance estimator and a spherical-coordinate quantile envelope, with a union-bound budget0.01 over all predeclared row/center/bank pairs in each fit/catalog case. Candidate banks were fixed independently of the random rotation. Exact old cost is computed during numerical old-group preprocessing. This is an ideal-arithmetic finite-query statistical statement, not an authenticated integer certificate; this float64 implementation has no outward-rounding proof. The query-time estimate includes interval construction.','']
    if rabitq is None:
        lines.append('The RaBitQ numerical follow-up is prepared but has not completed; it supplies no result yet.')
    else:
        lines += ['| Data,128 rows | Label agreement % | Estimated-sign agreement % | Statistical decision coverage % | Wrong statistical decisions |',
                  '|---|---:|---:|---:|---:|']
        for r in summary:
            if r['n']!=128 or r['method']!='rabitq_raw':continue
            rr=[x for x in rabitq['records'] if x['dataset']==r['dataset'] and x['n']==128]
            lines.append(f'| {r["dataset"]} | {number(r["assignment_agreement"],100)} | {number(r["point_sign_agreement"],100)} | {number(r["statistical_coverage"],100)} | {sum(x["wrong_statistical_decision"] for x in rr)} |')
        lines.append('')
        lines.append('The observed interval-validity counts and individual request results are retained, including any failure. Statistical interval coverage is not included in the deterministic-certificate columns above.')
    lines.append('')
    lines += ['## Real-proof adapter gate','',
        'One pinned fit per source,32 rows and the first four training-only candidates, with every candidate retained. All arms use eight workers serially under the same laptop guards. The table uses enclosing measured invocation wall time; certificate totals additionally charge basis fitting. It includes orchestration/validation, not just prover time. Active-phase sums are retained separately in JSON. ZKML diagnostic MockProver time is excluded explicitly. Full and adapter exact public costs reveal more than the certificate output.','',
        '| Data | Method | Status | Minutes | Coverage before Full | Full fallback |',
        '|---|---|---|---:|---:|---:|']
    for r in proof:
        lines.append(f'| {r["dataset"]} | {r["method"]} | {"complete" if r["complete"] else "unmeasured/incomplete"} | {number(r.get("seconds"),1/60,3)} | {str(r["coverage"])+"/"+str(r["requests"]) if r["complete"] else "N/A"} | {r.get("fallback","N/A")} |')
    lines += ['','## Equal cost-aware warm comparison','',
        'A separate engineering follow-up applies the existing routing policy equally to centroid, PCA and both projection adapters. Cost calibration is frozen from the earlier seed521 Q8 pilot, not tuned to this new cohort. Each known batch contains four fixed requests; the whole native preflight is charged before batch activation. Authenticated rows, applicable old-cost proofs and setup are reused; all changing-request proofs and public replay are fresh. This is a warm phase-wall comparison, not independently measured RPC latency or a cold crossover.','',
        'One newly measured warm Full arm is shared per dataset. It is not counted as four baseline repetitions. Copy/export and independent integer oracle checks are outside these phase sums. Missing cells establish no ratio.','',
        '| Data | Basis / policy | Compact requests | Direct Full | Minutes | Warm Full min. | Full / service |',
        '|---|---|---:|---:|---:|---:|---:|']
    if not warm:
        lines.append('| — | Prepared; not yet measured | N/A | N/A | N/A | N/A | N/A |')
    for r in warm:
        lines.append(f'| {r["dataset"]} | {LABELS[r["method"]+"_certificate"]} | {r["certificate_requests"]} | {r["direct_full_requests"]} | {number(r["measured_warm_routed_seconds"],1/60,3)} | {number(r["shared_warm_full_seconds"],1/60,3)} | {number(r["shared_full_over_routed"],decimals=3)} |')
    lines.append('')
    lines.append(f'Equal-routing warm follow-up complete: **{warm_complete}**. The initial unexecuted warm plan is retained; the active v2 plan uses portable dependency paths. The case list, method policy and calibration are unchanged.')
    lines += ['',f'Proof queue status: **{queue.get("status","not started")}**. Missing or stopped cells establish no speed ratio.', '',
        'Raw/projection algorithms use optimized floating-point sketches and coresets use exact integer sampled distances plus floating weights; our native certificates use conservative arbitrary-precision integer arithmetic. Native speed comparisons therefore concern these explicit implementations, not a universal mathematical lower bound.','',
        '## Applicability of other papers','',
        '- ezDPS proves a DWT–PCA–SVM inference pipeline with model privacy. Its original classification accuracy and runtime are not measurements of our private-catalog, original-space cost-change task. No imported speed ratio is claimed.',
        '- Zhang et al.2021 uses secure multiparty computation for outsourced clustering. It has a different statement and trust/interaction contract; no same-task implementation or runtime is claimed here. The frozen initial source note\'s word "encryption" was overbroad; this description follows the publisher\'s abstract.',
        '- ZKML is retained as a separately measured primitive adapter. This does not benchmark its complete optimizing compiler.','',
        '## Validation and environment','',
        'All reported certificate-adapter proof cells passed their independent exact-integer oracle and public verifier checks, including paid fallback. No wrong deterministic certified decision was observed in the native gate. These checks establish correctness on the recorded cases, not a population guarantee.','']
    if tests is not None:
        lines.append(f'Targeted Linux test receipt: complete={tests["complete"]}, exit={tests["returncode"]}. Command: `{" ".join(tests["command"][1:])}`. Retained log summary: `{test_summary}`. It covers baseline formulas, overflow handling, valid intervals, statistical/deterministic report separation, routing and learning workflows.')
        lines.append('')
    if environment is not None:
        lines.append(f'Measurement host: {", ".join(environment["cpu_models"])}; {environment["platform"]}; Python {environment["python"].split()[0]}. Proof receipts record eight workers, affinity0–7 and process-tree peak RSS. Numerical BLAS work uses one thread. Pytest was installed into the isolated validation environment after cold measurements; numerical package versions were unchanged. No server or GPU was used.')
        lines.append('')
    lines += [
        '## Reproduction','',
        'Use a fresh root inside this checkout for new timing measurements. The existing gate retains its immutable plan, row IDs, model inputs, source pins, all attempts, public proof bundles and exact integer checks.','',
        '```sh','PYTHONPATH=code OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.published_baseline_gate prepare --root code/validation/outputs/fresh_gate',
        'PYTHONPATH=code OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.published_baseline_gate native --root code/validation/outputs/fresh_gate',
        '# Prepare the follow-ups after the first native gate, before their own outcomes:',
        'PYTHONPATH=code python -m experiments.rabitq_baseline_gate prepare --root code/validation/outputs/fresh_gate',
        'PYTHONPATH=code python -m campaigns.published_baseline_warm_gate prepare --root code/validation/outputs/fresh_gate',
        '# Real proofs require the pinned Linux binaries and the guarded laptop wrapper:',
        'PYTHONPATH=code python -m experiments.published_baseline_gate proofs --root code/validation/outputs/fresh_gate',
        'PYTHONPATH=code python -m experiments.rabitq_baseline_gate run --root code/validation/outputs/fresh_gate',
        'ln -s ../STOP code/validation/outputs/fresh_gate/warm_followup_v2/STOP',
        'PYTHONPATH=code python -m campaigns.published_baseline_warm_gate run --root code/validation/outputs/fresh_gate',
        'PYTHONPATH=code python -m analysis.published_baseline_gate_report --root code/validation/outputs/fresh_gate --stem fresh_gate_report','```','']
    (REPO/'output'/f'{stem}.md').write_text('\n'.join(lines),encoding='utf-8')
    source_files=[REPO/'code/commitjl/published_audit_baselines.py',REPO/'code/commitjl/rabitq_audit_reference.py',
        REPO/'code/experiments/published_baseline_gate.py',REPO/'code/experiments/rabitq_baseline_gate.py',
        REPO/'code/campaigns/published_baseline_warm_gate.py',Path(__file__)]
    source_files += [root/'plan.json',root/'plan_pin.json',root/'native_results.json']
    if rabitq_path.exists():source_files.append(rabitq_path)
    for subdir in ['rabitq_followup','warm_followup_v2']:
        source_files += [p for name in ['plan.json','plan_pin.json','matched_queue.json','checks.json']
                         if (p:=root/subdir/name).exists()]
    source_files += [p for name in ['environment.json','validation_environment.json','resources/tests.attempt_01.json','resources/tests.attempt_01.log']
                     if (p:=root/name).exists()]
    source_files += sorted((root/'inputs').glob('*/pin.json'))
    source_files += [REPO/'code/analysis/published_baseline_gate_figures.py']
    report_files=[REPO/'output'/f'{stem}{suffix}' for suffix in ['.md','.json','_native_per_fit.csv','_native_requests.csv']]
    report_files += [p for suffix in ['_summary.png','_summary.svg']
                     if (p:=REPO/'output'/f'{stem}{suffix}').exists()]
    save(REPO/'output'/f'{stem}_receipt.json',dict(complete=value['complete'],
        paper_and_methodology_unchanged=True,
        source_sha256={p.relative_to(REPO).as_posix():sha(p) for p in source_files},
        report_sha256={p.relative_to(REPO).as_posix():sha(p) for p in report_files},
        proof_attempt_receipts_sha256={p.relative_to(REPO).as_posix():sha(p) for p in sorted((root/'proofs/resources').glob('*.json'))},
        warm_attempt_receipts_sha256={p.relative_to(REPO).as_posix():sha(p) for p in sorted((root/'warm_followup_v2/resources').glob('*.json'))}))
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--stem',default='published_baseline_gate_20261029')
    args=parser.parse_args();result=report(args.root.resolve(),args.stem)
    print(json.dumps(dict(native_rows=len(result['native_summary']),proof_rows=len(result['proof_rows']),proof_complete=result['proof_campaign_complete'])))
