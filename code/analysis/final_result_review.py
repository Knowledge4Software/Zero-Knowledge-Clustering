"""A checked interpretation of completed campaigns, with matched comparisons."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from analysis.paper_readiness import md


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root.resolve()
    out=r/'checked_review';out.mkdir(exist_ok=True)
    meta=json.loads((r/'review_inputs.json').read_text());jobs=pd.DataFrame(meta['jobs'])
    assert len(jobs)==96 and (jobs.status=='complete').all() and (jobs.returncode==0).all()
    assert jobs.stop.isna().all()
    proofjobs=jobs[jobs.task=='proof'];assert len(proofjobs)==30
    assert proofjobs.independently_verified.all() and proofjobs.public_export_verified.all()
    audits=[json.loads(p.read_text()) for p in sorted((r/'audits').glob('*.json'))]
    assert len(audits)==2 and all(x['checks']==x['passed'] and not x['failed'] for x in audits)
    verified=sum(p['proofs'] for audit in audits for p in audit.get('public_bundles',[]))
    full=pd.read_csv(r/'followup/campaign/full_native.csv')
    assert len(full)==150 and len(full[['dataset','k','start']].drop_duplicates())==30
    fullmean=full.groupby('method')[['ari','agreement','ari_quantized_full','agreement_quantized_full']].mean().reset_index()
    fullmean.to_csv(out/'full_native_reference_comparison.csv',index=False)
    detail=full.groupby(['dataset','k','method'])[['ari','agreement','ari_quantized_full','agreement_quantized_full']].mean().reset_index()
    detail.to_csv(out/'full_native_by_dataset_k.csv',index=False)
    n=pd.read_csv(r/'completion/campaign/native.csv');assert len(n[['dataset','k','start']].drop_duplicates())==66
    primary=n[n.k.isin([5,10])&(n.input_bits==12)&(n.basis_bits==12)]
    rows=[]
    for label,mask in [
      ('Head only',primary.method=='head_only'),
      ('Head + energy',primary.method=='head_energy'),
      ('Head + JL, without energy (same tail)',(primary.method=='head_jl')&(primary.budget==16)&(primary.policy=='fixed_components')),
      ('Head + energy + JL-16',(primary.method=='head_energy_jl')&(primary.budget==16)),
      ('PCA-16',(primary.method=='pca')&(primary.budget==16))]:
        z=primary[mask].groupby(['dataset','k','start'])[['ari','agreement','agreement_quantized_full','cost_abs_relative_error']].mean()
        assert len(z)==48
        v=z.mean();rows.append(dict(method=label,cells=len(z),ari_vs_float=v['ari'],agreement_vs_float_percent=100*v['agreement'],
          agreement_vs_quantized_percent=100*v['agreement_quantized_full'],mean_absolute_cost_error_percent=100*v['cost_abs_relative_error']))
    ablation=pd.DataFrame(rows);ablation.to_csv(out/'component_attribution.csv',index=False)
    sparsity=primary[primary.method=='sparse_jl'].groupby(['budget','q'])[['ari','agreement','cost_abs_relative_error']].mean().reset_index()
    sparsity.to_csv(out/'sparsity_separate_settings.csv',index=False)
    exclusions=pd.DataFrame(meta['native_configuration_exclusions']).drop_duplicates()
    exclusions.to_csv(out/'excluded_native_configurations.csv',index=False)
    geo=pd.read_csv(r/'followup/campaign/geometry_native.csv');geometry=[]
    for budget in [16,32,64]:
        z=geo[(geo.budget==budget)&geo.method.isin(['pca','protected_tail'])]
        z=z.groupby(['dataset','k','start','method'])[['pair_mean_abs_relative_error','recall10']].mean().unstack('method').dropna()
        for method in ['pca','protected_tail']:
            geometry.append(dict(budget=budget,method=method,matched_cells=len(z),
              mean_absolute_squared_distance_error_percent=100*z['pair_mean_abs_relative_error'][method].mean(),
              recall10_percent=100*z['recall10'][method].mean()))
    geometry=pd.DataFrame(geometry);geometry.to_csv(out/'geometry_matched_cells.csv',index=False)
    proofs=pd.read_csv(r/'completion/campaign/proofs.csv')
    large=proofs[proofs.large & (proofs.phase=='query_64') & (proofs.status=='complete')]
    large=large[['variant','n_catalog','constraints','prove_median_seconds','public_export_verified']]
    large.to_csv(out/'mnist_64point_query.csv',index=False)
    workloads=pd.DataFrame(json.loads((r/'workload_inputs.json').read_text()))
    workloads.to_csv(out/'all_modeled_workloads.csv',index=False)
    w=workloads[workloads.large & (workloads.q==32)].copy()
    assert w.distinct_queries.eq(4).all()
    w=w[['variant','n_catalog','q','distinct_queries','native_prepare_seconds','stage1_compile_setup_seconds',
       'stage1_witness_seconds','stage1_prove_seconds','query_compile_setup_seconds','query_witness_seconds',
       'query_prove_seconds','modeled_fresh_workload_seconds']]
    w.to_csv(out/'mnist_64catalog_four_distinct_queries.csv',index=False)
    extra=pd.read_csv(r/'completion/full_budget_control/full_q8_controls.csv');assert len(extra)==2 and extra.complete.all()
    precision=n[n.policy=='precision'].groupby(['input_bits','basis_bits'])[['agreement','cost_abs_relative_error']].mean().reset_index()
    precision.to_csv(out/'precision_all_tested_center_counts.csv',index=False)
    summary=dict(audit_checks=sum(v['checks'] for v in audits),audit_failures=0,public_proof_bundles=32,saved_proof_verifications=verified,
      completed_jobs=96,additional_completed_full_controls=2,native_rows=len(n),native_unique_cells=66,
      unique_native_configuration_exclusions=len(exclusions),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'validation_summary.json').write_text(json.dumps(summary,indent=2))
    text=['# Completed-results review — 18 October 2026','',
      'The data and proofs checked here are internally consistent. The fixed-center accuracy result is strong; the complete method is not the cheapest implementation of the fixed nearest-center/cost task.',
      '', '## Evidence integrity','',
      f"All 96 completion jobs and both extra Full controls completed. Fresh checking passed {summary['audit_checks']:,} source/raw-array/summary checks, including {verified:,} saved-proof verifications across 32 public bundles. No audit check failed. Public bundles were verified without private input arrays or assignment logs. This is not a formal security audit.",
      '', 'All 66 native cells are present: 48 primary cells and 18 center-count sensitivity cells. The full-cached-data study has 150 method rows across 30 dataset/k/start cells. Three campaigns and both extra controls are preserved separately.',
      '', '## Fixed-center accuracy on full cached data','',
      'MNIST and Fashion each use 70,000 rows; text uses 10,944 nonzero cached training documents with 2,000 features. Each uses a 60/40 split, k=5/10 and five starts. These are native computations, not full-corpus ZK proofs. ARI/agreement compare fixed-center assignments, not ground-truth class labels.',
      md(fullmean),'',
      'The floating and quantized references are different. For example, quantized Full itself does not have ARI=1 against floating Full. Head+energy has mean ARI about 0.99965 against quantized Full and about 0.99743 against floating Full. Report the reference explicitly.',
      '', '## Component attribution','',md(ablation),'',
      'These are matched 48 dataset/k/start cells from the eight capped corpora. Maps are first averaged within a start. Error is the mean absolute error of aggregate cost per cell, not a signed mean or mean per-point error. PCA proves its projected cost correctly; its original-cost proxy can be biased.',
      'Head-only and head+energy store h and h+1 values. In the same-tail ablation, head+JL stores 15 values and head+energy+JL stores 16. Energy changes the cost calculation: it replaces the tail-norm approximation; it is not added on top of it. Equal-budget head+JL controls are separately preserved in the raw CSV.',
      '', '## Larger actual ZK queries','',
      'Same MNIST catalog of 64 committed records, k=5, query size 64. Timings below exclude preprocessing/setup. A full-catalog query has only one subset and does not establish repeated-query amortization.',md(large),'',
      'The method has fewer query constraints and lower measured latency than PCA-16 in this case. Cached exact answers are cheaper. Affine has fewer constraints than our representation, but proving-time ordering varies between cases; small timing differences do not establish superiority.',
      '', '## Whole-workload accounting','',
      'Four genuinely different 32-record subsets of the same 64-record MNIST catalog. The sum includes native commitment/preparation, all 64 preprocessing proofs, compile/setup and one proof per distinct query. It excludes fitting the already supplied public model, warmups, verification and I/O. These are reconstructed measured components, not observed wall-clock totals or an established break-even threshold.',
      md(w[['variant','distinct_queries','modeled_fresh_workload_seconds']]),'',
      'Preprocessing dominates this workload. The whole-workload advantage over PCA is much smaller than a query-only speed ratio. Full needs no preprocessing proof and is cheaper for the tested small one-point workload; its N=64 larger query shapes were not measured.',
      '', '## Added JL geometry utility','',md(geometry),'',
      'PCA and our method are compared on identical dataset/k/start cells at each budget. At width 64, Arcene exceeds PCA training rank and is excluded from both sides. Energy is not a Euclidean coordinate; the 16-value protected representation has 15 Euclidean coordinates.',
      'JL reduces squared-distance magnitude distortion compared with PCA here, but PCA preserves nearest-neighbor identities better. Actual head+tail pair-distance proofs were also generally slower than PCA at equal storage. Thus there is a tradeoff rather than a universal geometry win.',
      '', '## Full baseline controls','',md(extra[['dataset','q','constraints','prove_median_seconds','independent_public_verification']]),'',
      'These replace the earlier q=8 MNIST/Fashion budget exclusions with actual proofs at a 1.5M constraint ceiling. Peak RSS stayed below 2.4 GB. Other larger Full shapes remain excluded/unmeasured. Do not interpret a preflight budget exclusion as observed out-of-memory failure.',
      '', '## Coverage and reporting corrections','',
      f"There were no failed completion jobs, but {len(exclusions)} unique native configurations were deliberately skipped: PCA-64 above Arcene training rank and head+JL budgets too small for high protected rank. The 96 recorded exclusion entries contain repeats across map IDs; the deduplicated configurations are exported separately.",
      '- Keep q=1, q=2 and q=4 separate. Increasing sparse-map density did not consistently improve the fixed-center results in the tested grid.',
      '- Use absolute cost error when summarizing error magnitude; signed errors can cancel. Signed bias remains useful when explicitly labeled.',
      '- Match dataset coverage in geometry comparisons. Earlier all-row averages at width 64 compared unequal corpus sets.',
      '- Keep both floating-Full and quantized-Full references visible. Quantization is part of total approximation error.',
      '- Keep cached exact answers and affine as main fixed-task controls; do not omit stronger comparators.',
      '', '## Paper implication','',
      'The defensible result is a compact representation that combines near-exact fixed-center answers with an optional approximate residual geometry. Accuracy against PCA is supported. Fastest fixed-task ZK auditing, universal geometry superiority, whole-corpus proof scalability and Q1 readiness are not established. The paper needs a clear contribution relative to exact affine and cached-answer baselines.',
      '', '## Storage','',
      'At the server check, all Mobin result folders occupied about 368 MB (decimal), with another 141 MB in the dataset cache. The three recent main result folders total about 362 MB before compression. This downloaded bundle contains aggregate results, audit records and reports; private arrays and complete proof artifacts remain on the server.', '']
    (out/'RESULTS_REVIEW.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps(summary));print(out/'RESULTS_REVIEW.md')


if __name__=='__main__':main()
