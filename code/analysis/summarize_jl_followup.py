"""Check and describe the second JL exploration without proof-speed claims."""
from pathlib import Path
import argparse,csv,json
from commitjl.pilot_io import save,sha,sign


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--lower-rank',type=Path);a=p.parse_args();root=a.root.resolve();summary=json.loads((root/'summary.json').read_text());assert summary['complete']
    predictions=list(csv.DictReader((root/'prediction.csv').open()));certs=list(csv.DictReader((root/'certificates.csv').open()));assert len(predictions)==1120 and len(certs)==126
    for r in certs:
        truth=int(r['truth']);lo,hi=int(r['lower']),int(r['upper']);den=int(r['denominator'])
        assert lo<=truth*den<=hi and r['decision']==sign(lo,hi) and r['decision'] in ['inconclusive',sign(truth,truth)]
    rows=summary['results'];pred=[r for r in rows if r['idea']=='prediction'];basis=[r for r in rows if r['idea']=='basis'];ref=[r for r in rows if r['idea']=='refinement']
    for r in pred:
        g=[v for v in predictions if v['variant']==r['variant']];assert len(g)==r['requests']==160
        assert sum(int(v['original_correct'])-int(v['correct']) for v in g)==r['additional_errors']
    hashes={name:sha(root/name) for name in ['protocol.json','summary.json','prediction.csv','certificates.csv']}
    save(root/'checked_receipt.json',dict(complete=True,prediction_rows=1120,checked_intervals=126,all_intervals_contain_truth=True,false_conclusive=0,new_proofs=0,sha256=hashes))
    lines=['# Further JL experiments, 25 October 2026','',
        'All fixed variants/seeds and negative results are retained. The ensemble compares every160 stored head update; this reuses development fits. The covariance/sketch and refinement screens use all6 previously difficult natural regressions, a post-hoc stress set. Neither is new independent validation. No production predictor or proof binary changed.','',
        '## Assignment estimates','',
        '| Variant | Mean agreement (%) | Worst update (%) | Extra errors vs original | Extra sketch scalars/row |','|---|---:|---:|---:|---:|']
    for r in pred:lines.append(f"| {r['variant']} | {r['agreement_percent']:.6f} | {r['minimum_agreement_percent']:.6f} | {r['additional_errors']} | {r['stored_jl_values_per_row']} |")
    lines+=['',f"The original predictor mean is{pred[0]['original_agreement_percent']:.6f}%. The three-sketch shrunken estimate removes only3 errors among163840 record/update comparisons; its worst update is lower than the original worst. This tiny development gain does not establish a useful improvement or justify adding its48 sketch scalars and authentication cost. Margin gating does not produce a net accuracy improvement. All variants retain the separate no-JL rounded-Gram control in the raw exports.", '',
        '## Certificate subspaces','',
        '| Variant | Difficult regressions certified |','|---|---:|']
    for r in basis:lines.append(f"| {r['variant']} | {r['conclusive']}/{r['requests']} |")
    lines+=['','Covariance sketches are sparse random range finders with zero/one/two training-covariance multiplications. One/two powers use training data only and reuse the unchanged deterministic rounded-basis bounds. This is randomized subspace iteration, not a new JL distortion theorem. All variants have the same79 certificate rank; power methods use8 oversampling columns before selecting16 directions. Fitting times are retained per fit. A matching decision count does not imply tighter intervals or a faster prover.','',
        '## Exact-correction scheduling estimates','',
        '| Certificate/hint | Decisions | Total exact rows | Maximum per request |','|---|---:|---:|---:|']
    for r in ref:lines.append(f"| {r['variant']} | {r['conclusive']}/{r['requests']} | {r['total_exact_rows']} | {r['max_exact_rows']} |")
    lines+=['','Exact rows here are numerical oracle refinements selected only after an untrusted sketch/width ordering. Certificate acceptance still checks deterministic intervals. An authenticated selective exact-replacement proof circuit is not implemented; no faster-proof claim follows. Compare equal sketch budgets: median3x16 stores48 values, single64 stores64, median3x64 stores192.','',
        f"All126 final interval rows were independently rechecked against recorded integer Full truth. Zero incorrect conclusive decisions; zero new SNARKs. Total numerical run:{summary['elapsed_seconds']/60:.2f}minutes. Source/file hashes are in checked_receipt.json and the run protocol. Artifacts remain private for supervisor review."]
    if a.lower_rank:
        lower=a.lower_rank.resolve();extra=json.loads((lower/'summary.json').read_text());assert extra['complete'];r=list(csv.DictReader((lower/'results.csv').open()));assert len(r)==126
        for v in r:
            lo,hi,den,truth=map(int,[v['lower'],v['upper'],v['denominator'],v['truth']]);assert lo<=truth*den<=hi and v['decision']==sign(lo,hi)
        save(lower/'checked_receipt.json',dict(complete=True,checked_intervals=126,new_proofs=0,sha256={n:sha(lower/n) for n in ['protocol.json','summary.json','results.csv','fitting.json']}))
        lines+=['','## Lower-rank follow-up','',
            'This post-hoc extension keeps all6 difficult cases and all three seeds. It compares training-aware sparse sketches against the trained residual-SVD control at each equal rank.','',
            '| Rank | Trained SVD | JL, one covariance multiplication (three seeds) | JL, two covariance multiplications (three seeds) |',
            '|---:|---:|---|---|']
        for rank in [71,75,79]:
            control=next(x for x in extra['results'] if x['variant']==f'trained_r{rank}')['conclusive'];groups=[]
            for power in [1,2]:groups.append(', '.join(str(x['conclusive'])+'/6' for x in extra['results'] if x['variant'].startswith(f'jl_cov_r{rank}_p{power}_')))
            lines.append(f'| {rank} | {control}/6 | {groups[0]} | {groups[1]} |')
        lines+=['','Rank71 with two covariance multiplications matches4/6 from trained79, but trained71 already matches4/6 too. Thus this screen supports testing lower auxiliary rank, not attributing a unique gain to JL. A full stored-suite/independent-fit lower-rank sensitivity and matched proving-cost comparison is not established by these6 post-hoc cases. All126 additional intervals contain Full truth; zero new proofs.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(complete=True,checked_intervals=len(certs),report=str(root/'REPORT.md'))))


if __name__=='__main__':main()
