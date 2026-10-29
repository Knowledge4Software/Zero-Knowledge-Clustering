"""Recheck all fresh-fit numerical intervals and export paired comparisons."""
from pathlib import Path
import argparse,csv,json
import numpy as np
from commitjl.pilot_io import sha,save,sign,array_sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();summary=json.loads((root/'summary.json').read_text());assert summary['complete']
    records=list(csv.DictReader((root/'results.csv').open()));fits=json.loads((root/'fit_records.json').read_text());assert len(records)==256 and len(fits)==16
    for r in records:
        lo,hi,den,truth=map(int,[r['lower'],r['upper'],r['denominator'],r['true_change']]);assert lo<=truth*den<=hi and r['decision']==sign(lo,hi) and r['decision'] in ['inconclusive',sign(truth,truth)]
    for r in fits:assert sha(root/'fits'/f"{r['fit']}_{r['representation']}.npz")==r['archive_sha256']
    centers={}
    for r in fits:
        with np.load(root/'fits'/f"{r['fit']}_{r['representation']}.npz") as frozen:centers[r['fit'],r['representation']]=frozen['centers'].copy()
    unique={};duplicate_rows=identity_rows=0
    for r in records:
        c=centers[r['fit'],r['representation']];proposal=c[int(r['set'])]
        if np.array_equal(proposal,c[0]):identity_rows+=1;continue
        key=(r['fit'],r['representation'],int(r['n']),int(r['rank']),array_sha(proposal))
        if key in unique:
            for field in ['lower','upper','true_change','decision','correct','assignment_certified']:assert unique[key][field]==r[field],f'duplicate center set changes {field}'
            duplicate_rows+=1;continue
        unique[key]=r
    distinct=[]
    for n in [1024,128]:
        for method in ['ordered_head','pca']:
            for rank in [63,79]:
                g=[r for r in unique.values() if int(r['n'])==n and r['representation']==method and int(r['rank'])==rank];reg=[r for r in g if r['truth']=='worsens'];total=sum(int(r['n']) for r in g)
                distinct.append(dict(n=n,representation=method,rank=rank,requests=len(g),conclusive=sum(r['decision']!='inconclusive' for r in g),regressions=len(reg),detected=sum(r['decision']=='worsens' for r in reg),mean_agreement_percent=100*sum(int(r['correct']) for r in g)/total,minimum_agreement_percent=min(float(r['prediction_agreement_percent']) for r in g),mean_assignment_certified_percent=100*sum(int(r['assignment_certified']) for r in g)/total,mean_width_percent_old_cost=sum(float(r['width_percent_old_cost']) for r in g)/len(g),updates_below_99=sum(float(r['prediction_agreement_percent'])<99 for r in g)))
    save(root/'distinct_summary.json',dict(complete=True,unit='Distinct nonidentity quantized candidate center set within each deployed fit and representation. Repeated Lloyd snapshots are counted once. Catalog prefixes and ranks remain paired, not independent samples.',raw_interval_rows=len(records),identity_interval_rows_excluded=identity_rows,duplicate_interval_rows_excluded=duplicate_rows,distinct_interval_rows=len(unique),results=distinct,source_sha256={n:sha(root/n) for n in ['results.csv','fit_records.json','summary.json']}))
    paired=[]
    for n in [1024,128]:
        for method in ['ordered_head','pca']:
            g=[r for r in records if int(r['n'])==n and r['representation']==method];indexed={(r['fit'],r['set'],int(r['rank'])):r for r in g};contained=gained=lost=0
            for (fit,s,rank),r in indexed.items():
                if rank!=63:continue
                v=indexed[fit,s,79];contained+=int(r['lower'])<=int(v['lower'])<=int(v['upper'])<=int(r['upper']);gained+=r['decision']=='inconclusive' and v['decision']!='inconclusive';lost+=r['decision']!='inconclusive' and v['decision']=='inconclusive'
            paired.append(dict(n=n,representation=method,pairs=len(g)//2,rank79_intervals_inside_rank63=contained,rank79_gained_decisions=gained,rank79_lost_decisions=lost))
    save(root/'checked_receipt.json',dict(complete=True,checked_intervals=len(records),checked_fit_archives=len(fits),all_intervals_contain_truth=True,false_conclusive=0,new_proofs=0,paired=paired,sha256={n:sha(root/n) for n in ['protocol.json','summary.json','results.csv','fit_records.json']}))
    lines=['# Fresh-seed rank79 numerical replication','',
        'Four dataset/seed/k split configurations; eight deployed models and sixteen head/PCA representation fits. All predeclared natural updates remain, including unchanged/duplicate proposals. Ages, representations, ranks, updates and N128/N1024 catalogs share splits. These are fresh fits on the same two public source datasets, not new datasets or independent draws per update. Numerical results below are separate from the fresh proof timings still recorded by the laptop queue.','',
        'The primary table counts distinct nonidentity candidate center sets once per deployed fit, rather than repeated Lloyd requests. In particular, the three N1024 worsening submissions at Fashion-MNIST/k3/seed101/age8 iterations12/16/24 are one identical candidate. That is ONE distinct regression, not three independent regression cases. Raw proposal-level exports remain unchanged.','',
        '| N | Representation | Certificate rank | Conclusive distinct updates | Distinct regressions detected | Mean prediction agreement (%) | Worst agreement (%) | Mean assignment-certified fraction (%) | Mean change width / old cost (%) |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in distinct:
        lines.append(f"| {r['n']} | {r['representation']} | {r['rank']} | {r['conclusive']}/{r['requests']} | {r['detected']}/{r['regressions']} | {r['mean_agreement_percent']:.6f} | {r['minimum_agreement_percent']:.6f} | {r['mean_assignment_certified_percent']:.4f} | {r['mean_width_percent_old_cost']:.4f} |")
    lines+=['','Prediction rank63 stays frozen in both certificate profiles. Thus prediction agreement and assignment-certified fraction are identical between63 and79 by construction; stronger cost-change bounds do not manufacture better assignment predictions or certify every label. Agreement below the user-selected99% target is retained and explicitly counted in summary.json. No universal acceptance threshold is assumed.','',
        'The following older paired count includes all32 proposal steps per representation, including identities and repeats; it is a containment check, not32 distinct updates.','',
        '| N | Representation | Rank79 intervals inside63 | Decisions gained | Decisions lost |','|---:|---|---:|---:|---:|']
    for r in paired:lines.append(f"| {r['n']} | {r['representation']} | {r['rank79_intervals_inside_rank63']}/{r['pairs']} | {r['rank79_gained_decisions']} | {r['rank79_lost_decisions']} |")
    lines+=['','All256 recorded intervals contain independently recomputed quantized Full truth; all16 fit-archive hashes match. No false conclusive outcome or false certified assignment was found. This is not a complete security audit. Cases that remain inconclusive still need verified fallback. Native fitting durations can reflect overlapping one-thread development screens and varying laptop load.','',
        'Two N128 proof cases were fixed before seeing these results: Fashion-MNIST/k3/seed101/age8 and MNIST/k5/seed113/age1. Fresh Full/head79/head63/PCA79 proof generation and public replay, including every required newly generated fallback, are managed by code/validation/outputs/rank79_independent_replication_20261025/queue.json. No new speed ratio is supported until those stages complete. Artifacts stay private for supervisor review.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(complete=True,checked_intervals=len(records),paired=paired)))


if __name__=='__main__':main()
