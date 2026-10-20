"""Read a bound-audit snapshot without running or changing experiments."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from commitjl.sparse_gen import sparse_matrix_gen_csr


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def frame(path):
    return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()


def median(values):
    return float(np.median(values)) if values else 0.


def markdown(df):
    if df.empty:
        return '_Not yet available._'
    def fmt(v):
        if isinstance(v,(float,np.floating)):
            return f'{v:.6g}'
        return str(v).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |',
                      '| '+' | '.join(['---']*len(df.columns))+' |']+
                     ['| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])


def make_report(root,out):
    root=Path(root).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    status=read(root/'status.json')
    quality=frame(root/'native_quality.csv');geometry=frame(root/'native_geometry.csv')
    phases=frame(root/'phases.csv')
    checks=[];proofs=[];samples=[];pipelines=[];proof_quality=[];case_data={};models=[]
    def check(name,ok,detail=''):
        checks.append(dict(check=name,passed=bool(ok),detail=detail))
    for directory in sorted((root/'cases').glob('*')) if (root/'cases').exists() else []:
        if not (directory/'trusted.json').exists():continue
        trust=read(directory/'trusted.json');m=trust['model'];method=m['method'];ctx=read(directory/'native_context.json')
        case=directory.name;meta=dict(case=case,dataset=ctx['dataset'],k=m['k'],d=m['d'],n_catalog=m['n'],method=method)
        if m['basis']:
            u=np.asarray(m['basis'],float)/m['basis_scale'];c=np.asarray(m['centers'],float)
            delta=c[1:]-c[0];residual=delta-(delta@u.T)@u
            models.append(dict(**meta,rank=len(u),gram_spectral_error=float(np.linalg.norm(u@u.T-np.eye(len(u)),2)),
                max_center_span_relative_error=float(np.max(np.linalg.norm(residual,axis=1)/np.maximum(np.linalg.norm(delta,axis=1),1e-30)))))
        if method=='protected_tail':
            path=root/f'{ctx["dataset"]}_k{m["k"]}_challenge.json'
            challenge=read(path)
            params=hashlib.sha256(json.dumps(dict(centers=m['centers'],basis=m['basis'],n=m['n'],d=m['d']),sort_keys=True).encode()).hexdigest()
            seed=hashlib.sha256(f'bound-tail-v1/{trust["original_root"]}/{params}/{challenge["nonce"]}'.encode()).digest()
            _,a=sparse_matrix_gen_csr(seed,m['tail_width'],m['d'],1);a=a.tocsc()
            check(case+'/challenge_parameters',params==challenge['parameters'] and challenge['original_root']==trust['original_root'])
            check(case+'/canonical_challenge_matrix',seed.hex()==m['seed']==challenge['seed'] and
                  np.all(np.diff(a.indptr)==1) and a.indices.tolist()==m['tail_rows'] and a.data.astype(int).tolist()==m['tail_signs'])
        prep=read(directory/'prepare.json');s1=read(directory/'stage1.json') if (directory/'stage1.json').exists() else None
        independent=read(directory/'verify.json') if (directory/'verify.json').exists() else None
        queries=[dict(read(p),requested_q=int(p.stem.removeprefix('query_')))
                 for p in sorted(directory.glob('query_*.json')) if p.stem.removeprefix('query_').isdigit()]
        verified=bool(independent and independent['complete'])
        if independent:check(case+'/saved_bundle_verified',verified)
        check(case+'/owner_pin',trust['original_root']==prep['statement']['cx'])
        # Verify the archived key bytes still match the trusted setup pins.
        for key,want in trust['verifying_key_sha256'].items():
            path=directory/'keys'/f'{key}.vk'
            check(case+'/key/'+key,path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==want)
        if s1 and method!='full' and s1['complete']:
            check(case+'/stage1_coverage',s1['verified']==m['n'] and [s['id'] for s in s1['samples']]==list(range(m['n'])))
        for phase in ([s1] if s1 else [])+queries:
            ss=phase.get('samples') or []
            measured=[median(s['prove_seconds']) for s in ss]
            verification=[median(s['verify_seconds']) for s in ss]
            row=dict(**meta,phase=phase['phase'],q=phase.get('requested_q',0),
                status=phase['status'],complete=phase['complete'],independent_bundle_verified=verified,
                constraints=phase['constraints'],compile_seconds=phase['compile_seconds'],setup_seconds=phase['setup_seconds'],
                witness_total_seconds=sum(s['witness_seconds'] for s in ss),
                prove_sum_of_sample_medians_seconds=sum(measured),prove_median_seconds=median(measured),
                verify_median_seconds=median(verification),samples=len(ss),
                proof_bytes_median=median([s['proof_bytes'] for s in ss]),
                warmup_seconds=phase.get('warmup_seconds',0.),observed_phase_wall_seconds=phase['wall_seconds'])
            proofs.append(row)
            if phase['phase']=='query' and ss:
                check(f'{case}/query{row["q"]}/sample_shape',all(len(s['indices'])==row['q'] for s in ss))
            for sample in ss:
                samples.append(dict(**meta,phase=phase['phase'],q=row['q'],id=sample['id'],
                    indices=json.dumps(sample.get('indices',[])),psi=sample.get('psi',''),
                    assignments=json.dumps(sample.get('assign',[])),
                    witness_seconds=sample['witness_seconds'],native_answer_seconds=sample.get('native_answer_seconds',0.),
                    prove_seconds=median(sample['prove_seconds']),verify_seconds=median(sample['verify_seconds']),
                    proof_bytes=sample['proof_bytes'],repetitions=len(sample['prove_seconds']),
                    independent_bundle_verified=verified))
            if phase['phase']!='query' or not phase['complete']:continue
            # Diagnostic assignment metadata are emitted by the prover harness;
            # assignments are private witnesses, not public proof outputs.
            x=np.asarray(ctx['original_float_points']);centers=np.asarray(ctx['original_float_centers'])
            ari=[];agree=[];error=[];costratios=[]
            for sample in ss:
                ids=sample['indices'];xa=x[ids]
                dist=((xa[:,None,:]-centers[None,:,:])**2).sum(2)
                ref=dist.argmin(1);pred=np.asarray(sample['assign'],dtype=int)
                phi=float(dist[np.arange(len(ids)),ref].sum())
                denominator=ctx['input_scale']**2*(m['basis_scale']**4 if method in ['pca','protected_energy','protected_tail'] else 1)
                claimed=int(sample['psi'])/denominator
                ari.append(adjusted_rand_score(ref,pred));agree.append(float(np.mean(ref==pred)))
                error.append(claimed/phi-1);costratios.append(float(dist[np.arange(len(ids)),pred].sum()/phi))
            if ss:
                proof_quality.append(dict(**meta,q=row['q'],queries=len(ss),ari_mean=float(np.mean(ari)),
                    agreement_mean=float(np.mean(agree)),r_phi_mean=float(np.mean(costratios)),
                    claimed_original_cost_error_median=float(np.median(error)),
                    claimed_original_cost_abs_error_max=float(np.max(np.abs(error))),
                    independent_bundle_verified=verified,note='Private assignment metadata; small catalog, ARI at q=1 is uninformative'))
        s1samples=(s1.get('samples') or []) if s1 else []
        s1core=(s1['compile_seconds']+s1['setup_seconds']+
                sum(s['witness_seconds']+median(s['prove_seconds']) for s in s1samples)) if s1 else 0.
        for phase in queries:
            ss=phase.get('samples') or []
            if not phase['complete'] or not ss:continue
            q=len(ss[0]['indices']);distinct=len({tuple(s['indices']) for s in ss})
            check(f'{case}/query{q}/distinct_subsets',distinct==len(ss))
            qprove=sum(s.get('native_answer_seconds',0.)+s['witness_seconds']+median(s['prove_seconds']) for s in ss)
            # One setup per query shape and one proof per distinct subset.
            # Excludes warmups, verification, serialization, and shared KMeans.
            base=ctx['model_fit_seconds']+prep['native_prepare_seconds']+s1core
            pipelines.append(dict(**meta,q=q,distinct_queries=distinct,independent_bundle_verified=verified,
                model_fit_seconds=ctx['model_fit_seconds'],native_prepare_seconds=prep['native_prepare_seconds'],
                stage1_compile_setup_witness_prove_seconds=s1core,
                query_compile_setup_seconds=phase['compile_seconds']+phase['setup_seconds'],
                query_witness_prove_seconds=qprove,
                modeled_workload_prover_seconds=base+phase['compile_seconds']+phase['setup_seconds']+qprove,
                observed_prepare_stage1_query_phase_wall_seconds=prep['wall_seconds']+(s1['wall_seconds'] if s1 else 0)+phase['wall_seconds'],
                note='Reconstructed one-shape workload excludes KMeans/warmup/verification/I-O; phase-wall sum includes measured overhead and repetitions'))
        case_data[(ctx['dataset'],m['k'],method)]=(prep,s1,queries,directory)
    paired={key[:2] for key in case_data}
    for dataset,k in sorted(paired):
        group={method:value for (name,kk,method),value in case_data.items() if name==dataset and kk==k}
        check(f'{dataset}/k{k}/shared_original_root',len({v[0]['statement']['cx'] for v in group.values()})==1)
        if all(method in group for method in ['protected_energy','protected_tail']):
            no=read(group['protected_energy'][3]/'statistics.json');tail=read(group['protected_tail'][3]/'statistics.json')
            check(f'{dataset}/k{k}/tail_ablation_statistics',[(s['head'],s['energy']) for s in no]==[(s['head'],s['energy']) for s in tail])
        def answers(method):
            return {tuple(s['indices']):(s['assign'],s['psi']) for p in group[method][2] if p['complete'] for s in p.get('samples') or []}
        for a,b in [('full','affine'),('protected_energy','protected_tail')]:
            if a not in group or b not in group:continue
            aa,bb=answers(a),answers(b);common=aa.keys()&bb.keys()
            if common:check(f'{dataset}/k{k}/{a}_equals_{b}',all(aa[ix]==bb[ix] for ix in common),f'{len(common)} paired distinct queries')
    if not quality.empty:
        check('native_finite',np.isfinite(quality.select_dtypes('number')).all().all())
        affine=quality[quality.method=='affine']
        check('native_affine_matches_quantized_full',bool((affine.agreement_quantized_full==1).all()))
        metrics=['ari','agreement','ari_quantized_full','agreement_quantized_full','r_phi','claimed_original_cost_error',
                 'model_fit_seconds','compression_seconds','downstream_seconds','stored_columns']
        native=quality.groupby(['dataset','k','method'],as_index=False)[metrics].mean()
        native.to_csv(out/'native_summary.csv',index=False)
        aggregate=quality.groupby('method',as_index=False)[metrics].mean()
    else:native=aggregate=pd.DataFrame()
    gs=geometry.groupby(['dataset','k','method'],as_index=False)[['q95','mean_error','exceed_015','pairs']].mean() if not geometry.empty else pd.DataFrame()
    frames=dict(proof_summary=pd.DataFrame(proofs),query_and_row_samples=pd.DataFrame(samples),pipeline_summary=pd.DataFrame(pipelines),
                proof_instance_quality=pd.DataFrame(proof_quality),geometry_summary=gs,model_quality=pd.DataFrame(models))
    for name,df in frames.items():df.to_csv(out/(name+'.csv'),index=False)
    validation=dict(status=status['status'],complete=status['complete'],checks=checks,
        all_checks_passed=all(c['passed'] for c in checks),native_rows=len(quality),geometry_rows=len(geometry),
        independently_verified_bundles=sum(bool(v[3].joinpath('verify.json').exists() and read(v[3]/'verify.json')['complete']) for v in case_data.values()),
        failures=status.get('failures',[]),exclusions=status.get('exclusions',[]),
        input_files_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.json') if out not in p.parents})
    if status['complete']:
        expected=30 if status['scope']=='main' else 5
        check('completed_campaign_bundle_coverage',len(case_data)==status['planned_bundles']==status['bundles_complete']==expected)
        check('completed_campaign_independent_verification',validation['independently_verified_bundles']==expected)
        check('completed_campaign_native_coverage',len(quality)==(240 if status['scope']=='main' else 5))
        validation['all_checks_passed']=all(c['passed'] for c in checks)
    validation['analysis_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    # Keep per-case rows: aggregating a partial campaign across different datasets
    # can accidentally create an unpaired speed comparison.
    ps=frames['proof_summary'];pq=frames['proof_instance_quality'];pipe=frames['pipeline_summary']
    proofview=ps[[c for c in ['dataset','k','method','phase','q','constraints','samples','prove_median_seconds','compile_seconds','setup_seconds','independent_bundle_verified'] if c in ps]] if not ps.empty else ps
    lines=['# Original-data binding and subset-query results','',f'Campaign: `{root.name}`. Status: **{status["status"]}**.',
        f'Host: {status["host"]}; {status["threads"]} threads; {status["memory_bytes"]/2**30:.2f} GiB RAM.',
        f'Independently verified bundles: {validation["independently_verified_bundles"]}. Native rows: {len(quality)}.',
        '', '**This is a new Poseidon2/fixed-point cohort; do not pool it with the manuscript’s older MiMC timings.**',
        'A running/partial snapshot is not final evidence. Exclusions are not completed measurements.',
        '', '## Method and accuracy','',
        'Protected head spans public center differences. Residual energy restores the missing norm approximately after basis rounding. '
        'The optional JL tail is unused for fixed-center assignment/cost. Affine stores k−1 exact scores and one norm and is exact relative to quantized Full. '
        'PCA has 16 stored coordinates; protected-tail has at most 16 total statistics; no-tail and affine use k. '
        'ARI and agreement here compare nearest fixed-center assignments, not newly fitted clustering in compressed space or ground-truth classification.',
        'PCA proofs establish projected-space cost. The original-cost error column evaluates that cost as an original-space proxy; a large discrepancy does not invalidate the PCA proof.',
        '', 'Native means across the observed dataset/k/start cells (paired interpretation requires the per-dataset CSV):','',markdown(aggregate),
        '', '## Actual proofs: preprocessing and queries','',markdown(proofview),
        '', 'Compile/setup occur once per model/shape; each Stage-1 row is proved and verified. Query timings are medians after the declared warmup. '
        'Serialized bundles are independently verified from pinned public model/root/key material. Full needs no Stage 1. '
        'Whole-catalog queries have one unique subset and cannot establish a repeated-query advantage.',
        '', '## Quality on proof instances','',markdown(pq),
        '', 'These are small catalogs. Saved assignments are diagnostic private-witness metadata, not public authenticated outputs. '
        'The cost and membership statements are verified; ARI for a single point is uninformative. '
        'Use the full held-out native study for quality conclusions.',
        '', '## Complete workload accounting','',markdown(pipe),
        '', 'The reconstructed one-query-shape workload includes model fitting, Go native preparation/commitment, all Stage-1 row proofs, '
        'and query compilation/setup/witness/proving. It excludes shared KMeans, warmups, verification and I/O. '
        'It is a descriptive sum of measured components, not an observed end-to-end wall clock or a demonstrated break-even point. '
        'Python native compression is reported separately and is not added again to Go preparation. '
        'A whole-catalog result can be cached, and all methods can cache repeated identical queries.',
        '', '## Geometry and the actual contribution of JL','',markdown(gs),
        '', 'Energy is excluded from Euclidean pair-distance features. No-tail measures head only; protected-tail adds a sparse residual map. '
        'One reproducibly seeded map per clustering start is tested; proof bundles use separate post-commit challenges. '
        'The geometry table does not measure those specific challenged matrices. The 16-value protected storage budget includes one energy, leaving 15 Euclidean coordinates. '
        'These are native diagnostics; authenticated pair-distance query proofs and a downstream geometry workload are not implemented.',
        '', '## Coverage and limits','',
        f'Invariant checks: {sum(c["passed"] for c in checks)}/{len(checks)} passed. '
        f'Failures: {len(status.get("failures",[]))}; resource exclusions: {len(status.get("exclusions",[]))}.',
        'The owner commits quantized rows. Model fitting, correct challenge derivation and trusted setup approval are outside the circuit. '
        'The experiment is not a formal cryptographic audit or a proof of novelty. Larger full-corpus proofs remain unmeasured.',
        '', 'Raw result JSON, source hashes, resource records, trusted manifests, verification keys and serialized proofs are preserved. '
        'See `validation.json` and the CSV exports for exact coverage.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if not validation['all_checks_passed']:raise RuntimeError('Result invariant failed; inspect validation.json')
    return validation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path);parser.add_argument('--out',type=Path)
    args=parser.parse_args();v=make_report(args.results,args.out or args.results/'analysis')
    print(json.dumps({k:v[k] for k in ['status','complete','all_checks_passed','native_rows','independently_verified_bundles']}))


if __name__=='__main__':main()
