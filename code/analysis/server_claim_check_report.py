"""Recompute the server audit findings from raw outputs; no paper claims pooled."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,r))+' |' for r in rows])


def build(root):
    root=Path(root);state=json.loads((root/'status.json').read_text())
    if not state['complete']:raise ValueError('Server audit is not complete')
    frame=pd.read_csv(root/'rows.csv')
    assert len(frame)==state['expected_cases']==160
    assert not frame.duplicated(['dataset','k','method','n']).any()
    assert frame.stop.isna().all() and frame.returncode.eq(0).all()
    actual=frame[frame.prove_seconds.notna()]
    assert len(actual)==80 and actual.verify_ok.eq(True).all()
    # Preserve setup and witness costs separately; proving-only time is not
    # the complete cost of a circuit or of the original-data audit.
    timing_rows=[]
    for row in actual.itertuples():
        folder=root/'cases'/f'{row.dataset}_k{row.k}_{row.method}_n{row.n}_cost_const'
        result=json.loads((folder/'result.json').read_text())
        assert result['verify_ok'] and len(result['prove_seconds_samples'])==3
        assert len(result['verify_seconds_samples'])==3
        timing_rows.append(dict(dataset=row.dataset,k=row.k,method=row.method,n=row.n,
            compile_seconds=result['compile_seconds'],setup_seconds=result['setup_seconds'],
            witness_seconds=result['witness_seconds'],proof_bytes=result['proof_bytes']))
    timings=pd.DataFrame(timing_rows)
    timings.to_csv(root/'setup_witness_timings.csv',index=False)
    pairs=0;proved_pairs=0
    for (ds,k,n),group in frame.groupby(['dataset','k','n']):
        a=group[group.method.eq('protected_energy')].iloc[0]
        b=group[group.method.eq('head_energy_no_jl')].iloc[0]
        assert a.constraints==b.constraints and a.original_cost_rel_error==b.original_cost_rel_error
        first=root/'cases'/f'{ds}_k{k}_protected_energy_n{n}_cost_const'
        second=root/'cases'/f'{ds}_k{k}_head_energy_no_jl_n{n}_cost_const'
        ra=json.loads((first/'result.json').read_text());rb=json.loads((second/'result.json').read_text())
        assert ra['psi']==rb['psi'] and ra['assign']==rb['assign']
        ia=json.loads((first/'instance.json').read_text());ib=json.loads((second/'instance.json').read_text())
        h=ia['assign_dims'];assert ib['w']==h+1
        assert [v[:h+1] for v in ia['z']]==ib['z']
        pairs+=1;proved_pairs+=int(not ia['compile_only'])
    controls=json.loads((root/'native_controls.json').read_text());assert len(controls)==10
    assert all(v['exact_integer_affine_distances_equal'] for v in controls)
    gap=json.loads((root/'stage1_gap_demonstration.json').read_text());assert gap['verified']
    test=json.loads((root/'go_tests.json').read_text());assert test['returncode']==0
    datasets=json.loads((root/'datasets.json').read_text());assert len(datasets)==5
    summary=actual.groupby(['k','method','n'],as_index=False)[['constraints','prove_seconds','verify_seconds',
        'original_cost_rel_error','agreement','peak_rss_bytes']].median()
    summary.to_csv(root/'proof_summary.csv',index=False)
    slopes=[]
    for keys,g in frame.groupby(['dataset','k','method']):
        g=g.sort_values('n')
        for a,b in zip(g.itertuples(),list(g.itertuples())[1:]):
            slopes.append(dict(dataset=keys[0],k=keys[1],method=keys[2],from_n=a.n,to_n=b.n,
                constraints_per_added_point=(b.constraints-a.constraints)/(b.n-a.n)))
    pd.DataFrame(slopes).to_csv(root/'measured_slopes.csv',index=False)
    # Apply the manuscript's pointwise confidence-bound rule, rather than
    # treating thresholds on means as equivalent. Fixed-corpus intervals only.
    from experiments.paper_compression import bootstrap_crossed
    native=pd.read_csv(root/'previous_results/native_combined.csv');eligible=[]
    for keys,g in native[native.method.isin(['pca','sparse','protected','protected_int'])].groupby(['dataset','k','method','m']):
        row=dict(zip(['dataset','k','method','m'],keys))
        for metric in ['ari','r_phi']:
            grid=g.pivot(index='map',columns='start',values=metric).to_numpy()
            assert np.isfinite(grid).all()
            mean,lo,hi=bootstrap_crossed(grid,draws=2000)
            row.update({metric+'_mean':mean,metric+'_low':lo,metric+'_high':hi})
        row['passes_ci_rule']=row['ari_low']>=.8 and row['r_phi_high']<=1.1;eligible.append(row)
    eligibility=pd.DataFrame(eligible);eligibility.to_csv(root/'rq1_ci_qualification.csv',index=False)
    ci16=eligibility[eligibility.m.eq(16)].groupby('method').passes_ci_rule.sum().astype(int).to_dict()
    quality=actual.groupby('method',as_index=False).agg(
        median_signed_cost_error=('original_cost_rel_error','median'),
        median_absolute_cost_error=('original_cost_rel_error',lambda a:np.abs(a).median()),
        maximum_absolute_cost_error=('original_cost_rel_error',lambda a:np.abs(a).max()),
        mean_agreement=('agreement','mean'))
    overhead=timings.groupby('method',as_index=False)[['compile_seconds','setup_seconds','witness_seconds']].median()
    attacks=pd.read_csv(root/'previous_results/adversarial_attack_20261017.csv')
    attack_summary=attacks.groupby('method',as_index=False)[['true_inflation_a10','detected_inflation_a10',
        'flip_success','honest_assignment_error']].mean()
    attack_summary.to_csv(root/'earlier_attack_summary.csv',index=False)
    lines=['# Server verification and control study — 17 October 2026','',
        'The fixed-center head+energy result reproduces, with lower measured Stage-2 query constraints than the tested PCA representation. '+
        'However, removing the JL tail gives the same answers and constraints, and the original-data preprocessing proof is absent. '+
        'The evidence supports a conditional fixed-center query result; it does not finalize an end-to-end secure JL speedup.','',
        f'Server {state["host"]}: {state["cpu_count"]} CPUs, {state["memory_bytes"]/2**30:.2f} GiB RAM. '+
        'Four compute threads, GOMAXPROCS=4, GOGC=25, GOMEMLIMIT=4GiB. The rebuilt proof executable matches the previous executable byte for byte.','',
        f'Completed {len(frame)} constant-center cost-query configurations across five datasets, k=5/10 and n=1/2/4/8. '+
        f'{len(actual)} configurations generated and verified proofs at n=1/8 (three measured repetitions after one warmup). '+
        'n=2/4 are compile-only. This is a reproducibility/control study, not a new geometric-seed campaign.','',
        table(['Dataset','Rows after zero filtering','Features','Training rows','Held-out pool','Proved batch sizes'],
            [[v['dataset'],v['n'],v['d'],int(.6*v['n']),v['n']-int(.6*v['n']),'1 and 8'] for v in datasets]),'',
        'Dataset caps are Gaussian 2,000, MNIST/Fashion-MNIST 5,000 each, and text/Gisette 3,000 each. '+
        'Zero-norm rows are removed before the deterministic 60/40 split. Only the first eight held-out points are used in this proof control; '+
        'this is not an audit of every training or held-out row. Centers use KMeans(n_init=5, random_state=0); '+
        'PCA and both protected representations are fitted/derived from training data and these public centers. '+
        'The widths are m=16 versus head+energy only (k values when the center span has rank k-1); no 512-width run was launched.','',
        '## Main finding: the measured query savings do not need JL','',
        'For fixed centers, let U span their differences from the first center c0. '+
        'The protected head is h(x)=U^T(x-c0), and e(x)=||x-c0||^2-||h(x)||^2. '+
        'Then ||x-cj||^2=||h(x)-h(cj)||^2+e(x). The same energy is added for every center, '+
        'so the head determines assignment and head+energy determines cost. '+
        'This identity explains the high fixed-center agreement; it does not establish preservation of arbitrary pairwise distances or fresh clustering quality.','',
        f'All {pairs} matched compiled pairs have identical constraints, assignments, and exact integer claimed costs with and without the JL tail. '+
        f'This includes {proved_pairs} matched pairs with actual verified proofs. The head+energy/no-tail input copies exactly the same quantized head and energy; it is not re-quantized to gain precision.','',
        table(['k','Method','Points','Constraints (median)','Prove s (median)','Verify ms','Agreement'],
            [[v.k,v.method,v.n,f'{v.constraints:,.0f}',f'{v.prove_seconds:.3f}',f'{1000*v.verify_seconds:.2f}',f'{v.agreement:.3f}'] for v in summary.itertuples()]),'',
        'Table entries are medians across five datasets at the indicated k and point count. Agreement is against the original-space fixed-center assignment on the selected held-out points, not against external labels or a fresh clustering fit.','',
        table(['Method','Median signed cost error','Median absolute cost error','Maximum absolute cost error','Mean per-config agreement'],
            [[v.method,f'{100*v.median_signed_cost_error:.6f}%',f'{100*v.median_absolute_cost_error:.6f}%',
              f'{100*v.maximum_absolute_cost_error:.6f}%',f'{v.mean_agreement:.6f}'] for v in quality.itertuples()]),'',
        'These errors compare the integer proof answer, rescaled to floating point, against the original floating-point cost. The proved integer relation is exact; the original-space approximation is affected by quantization. '+
        'The quality table aggregates the actual-proof configurations equally (including nested n=1/8 batches); it is not a pooled accuracy estimate on independent points.','',
        table(['Method','Compile s (median)','Setup s (median)','Witness s (median)'],
            [[v.method,f'{v.compile_seconds:.3f}',f'{v.setup_seconds:.3f}',f'{v.witness_seconds:.6f}'] for v in overhead.itertuples()]),'',
        f'All measured proofs have sizes {sorted(timings.proof_bytes.unique().tolist())} bytes. '+
        'Setup/compilation and witness generation are reported separately from proving. Stage-1 preprocessing and its proof are absent. No end-to-end speedup follows by omitting them. '+
        'The centers are constants in these Groth16 circuits, so changing centers changes the circuit and requires corresponding setup/key material.','',
        'The measurements support a center-specific sufficient-statistics result. They cannot attribute fixed-center assignment/cost savings to the unused sparse tail. Geometry queries may still need the tail, but are outside this cost-query proof benchmark.','',
        'In the current energy-query circuit CZA authenticates the head and CZR authenticates only the energy column; neither root authenticates the unused tail. '+
        'The Go control confirms that removing the tail leaves both roots unchanged. A geometry protocol using that tail needs its own authenticated binding; '+
        'the experiment comment that the tail stays committed should not be read as a property established by these two Stage-2 roots.','',
        '## Stage-1 trust boundary reproduced','',
        f'A deliberately replaced energy value, together with its newly computed compressed root, produced an accepted Stage-2 proof. '+
        f'The integer claimed cost changed from {gap["old_cost"]} to {gap["new_cost"]}. '+
        'This is expected behavior of the Stage-2 relation, not a hash collision or a forged proof under the old root. '+
        'A separate verified Stage-1 relation is required to reject a false assertion that this new compressed root represents the original committed X. '+
        'Existing fixed-root tamper-rejection tests and the new trust-boundary tests passed. No Stage-1 proof has been implemented or timed here.','',
        '## Native controls: exact information is sufficient without randomness','',
        table(['Dataset','k','True cost inflation','Known linear map reports','No-JL head+energy reports'],
            [[v['dataset'],v['k'],f"{v['true_cost_inflation']:.3f}x",f"{v['known_linear_map_claim_inflation']:.3f}x",f"{v['known_no_jl_head_energy_inflation']:.3f}x"] for v in controls]),'',
        'An attacker uses the kernel of a known protected linear map. Its linear-sketch claim stays unchanged, while an honestly derived exact energy reports the original-cost change even without a fresh random projection. '+
        'This is a native algebra/perturbation control, not a general cryptographic attack/security proof. The exact integer affine-distance baseline also matched direct integer distances in all ten tested dataset/k cases. It has not been benchmarked as a proof.','',
        '## Corrected earlier results','',
        'The server independently recomputed the saved native data and actual-proof filters; see previous_result_checks.json. '+
        'At m=16 the native mean-based counts are protected/integer protected 16/16, PCA 13/16 and sparse JL 2/16. '+
        f'The corresponding pointwise confidence-bound qualification counts are {ci16}. '+
        'RQ1 r_phi evaluates the chosen labels in the original space; it is not the error of the projected cost claimed by the proof. '+
        'Intervals resample map/start axes conditional on the fixed datasets and have only three starts; there is no multiplicity-adjusted or new-data confirmation. '+
        'Earlier actual-proof rows have PCA assignment agreement 1.0 and sparse agreement 0.6; compile-only outcomes must not be labeled actual proofs.','',
        '## Earlier attack study: narrower conclusions','',
        table(['Method','True inflation (mean)','Claimed inflation (mean)','Flip fraction (mean)','Honest error (mean)'],
            [[v.method,f'{v.true_inflation_a10:.3f}x',f'{v.detected_inflation_a10:.3f}x',
              f'{v.flip_success:.4f}',f'{v.honest_assignment_error:.4f}'] for v in attack_summary.itertuples()]),'',
        'These are recomputed earlier native perturbation results, not newly forged end-to-end proofs. '+
        'The protected attack implementation tests a linear head+tail sketch, not the energy construction. '+
        'Its prepublished protected variant also has zero assignment flips; the table cannot establish that fresh sampling is necessary for protected assignment. '+
        'The fresh-map attacker uses one independently guessed map, not a general optimal adversary. '+
        'Flip errors on perturbed points and honest errors on original points have different denominators and do not isolate attack advantage. '+
        'The Pisot implementation is explicitly an unverified internal surrogate; these results do not demonstrate an attack on the cited published construction.','',
        '## Limits and next work','',
        '- Specify and implement the complete original-data-to-statistics binding before claiming an end-to-end original-data audit. Bind original roots, centers/basis, any challenge-derived map, scales, ranges, quantization and both compressed roots; prove correct head and energy derivation.',
        '- Define meaningful varying queries and authenticated subsets; repeated identical static answers can otherwise reuse a proof.',
        '- Extend the measured no-JL control to a complete preprocessing-plus-query comparison, and add an actual exact affine-score/distance proof baseline. The affine check here is native only.',
        '- Separate exact integer relations from approximations to original floating-point data. Keep tiny proof shapes and estimated Full values explicit.',
        '- Count representation bits as well as columns: residual energy uses squared-scale units and a wider integer range than a coordinate. A width of 16 alone does not establish a fair storage-bit comparison.',
        '- The original twelve-thread environment and this four-thread environment differ. New timing values are a fresh server configuration and are not pooled with old timing samples.',
        '- This audit did not rerun the large native grids or claim stronger general clustering quality.','',
        'Raw evidence: cases/*/instance.json, result.json, resources.json and process.log; rows.csv; proof_summary.csv; measured_slopes.csv; native_controls.json; go_tests.log/json; previous_results/; status.json.','']
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    outputs=['report.md','proof_summary.csv','setup_witness_timings.csv','measured_slopes.csv','rq1_ci_qualification.csv','earlier_attack_summary.csv','rows.csv','status.json']
    result=dict(complete=True,cases=len(frame),actual_proof_configs=len(actual),equal_tail_no_tail_pairs=pairs,
        actual_proved_equal_pairs=proved_pairs,expected_stage2_changed_root_acceptance=True,stage1_implemented=False,
        ci_rule_counts_m16=ci16,report_builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        previous_input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'previous_results').glob('*.csv')},
        output_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in outputs})
    (root/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    build(parser.parse_args().root)
