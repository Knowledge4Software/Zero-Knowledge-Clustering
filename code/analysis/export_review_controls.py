"""Export completed native controls and captured Full timings to a NEW paper folder."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[2]


def read_csv(path):
    with path.open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--full-capture',required=True,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    pre=ROOT/'code/validation/outputs/review_certificate_preflight_20261020'
    budget=ROOT/'code/validation/outputs/review_certificate_budget_20261020'
    sources=[pre/'new_set_summary.csv',pre/'certificate_preflight.csv',pre/'manifest.json',
             budget/'certificate_budget_summary.csv',budget/'certificate_budget_cells.csv',
             budget/'protocol.json',budget/'fit_records.json',a.full_capture/'capture.json']
    records=[]
    for source in sources:
        dest=a.out/(source.parent.name+'_'+source.name)
        shutil.copy2(source,dest)
        records.append(dict(source=str(source.relative_to(ROOT) if source.is_relative_to(ROOT) else source),
                            exported=dest.name,sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    def tex_table(filename,caption,label,header,rows):
        text=['\\begin{table*}[tp]',f'\\caption{{{caption}}}\\label{{{label}}}',
              '\\centering\\small','\\begin{adjustbox}{max width=\\textwidth}',
              '\\begin{tabular}{@{}'+''.join('l' if i<2 else 'r' for i in range(len(header)))+'@{}}',
              '\\toprule',' & '.join(header)+r' \\',r'\midrule']
        text += [' & '.join(row)+r' \\' for row in rows]
        text += [r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}',r'\end{table*}']
        (a.out/filename).write_text('\n'.join(text)+'\n',encoding='utf-8')
    names={'learned_energy':'Learned + energy','pca_energy':'PCA + energy'}
    pct=lambda v:f'{100*float(v):.2f}'
    rows=[]
    for x in read_csv(pre/'new_set_summary.csv'):
        dataset='MNIST' if x['dataset']=='mnist' else 'CIFAR/ResNet'
        rows.append([dataset+' ('+x['n']+')',names[x['method']],pct(x['mean_agreement']),
                     pct(x['mean_certified_fraction']),pct(x['min_certified_fraction']),pct(x['mean_interval_relative_width'])])
    tex_table('native_certificates.tex',
              'Native integer certificate preflight, not proof timings. Means cover nine updated center sets; agreement is with quantized Full. Width is $(U-L)/L$. Worst coverage is over these nine sets. No updated set passes the illustrative 99\\% coverage and 1\\% width policy.',
              'tab:native-certificates',['Catalog ($N$)','Method','Agreement (\\%)','Certified (\\%)','Worst (\\%)','Width (\\%)'],rows)
    rows=[]
    for x in read_csv(budget/'certificate_budget_summary.csv'):
        if (int(x['rank'])==15 and int(x['basis_scale'])!=4096) or (int(x['rank'])!=15 and int(x['basis_scale'])!=16384):continue
        rows.append([x['rank']+'/'+x['basis_scale'],names[x['method']],pct(x['agreement']),
                     pct(x['certified_fraction']),pct(x['worst_certified_fraction']),pct(x['interval_relative_width'])])
    tex_table('native_budget.tex',
              'Native MNIST $N=1024$ budget controls (nine updated sets, quantized Full reference). All ranks and both basis scales are retained in the CSV artifact; this table shows the original configuration and larger-rank high-scale controls. These are not proof benchmarks.',
              'tab:native-certificate-budget',['Head rank / $T$','Method','Agreement (\\%)','Certified (\\%)','Worst (\\%)','Width (\\%)'],rows)
    fullrows=[]
    for summarypath in sorted(a.full_capture.glob('*/summary.json')):
        case=summarypath.parent
        summary=json.loads(summarypath.read_text())
        if not summary['complete']:continue
        protocol=json.loads((case/'protocol.json').read_text())
        prep=json.loads((case/'bundle/prepare.json').read_text())
        q=json.loads((case/f"bundle/full_reuse_{protocol['batch']}.json").read_text())
        replay=json.loads((case/'public_bundle/verify.json').read_text())
        assert q['complete'] and replay['complete'] and q['verified']==replay['verified']
        n=protocol['n'];batch=protocol['batch'];sets=protocol['center_sets']
        assert len(q['samples'])==n//batch*sets and protocol['setup_reused_across_centers']
        per_set=[]
        for s in range(sets):
            samples=sorted([x for x in q['samples'] if x.get('centre_set',0)==s],key=lambda x:x['id'])
            assert [i for x in samples for i in x['indices']]==list(range(n))
            timed=sum(x.get('native_answer_seconds',0.)+x['witness_seconds']+statistics.median(x['prove_seconds']) for x in samples)
            one=prep['native_prepare_seconds']+q['compile_seconds']+q['setup_seconds'] if s==0 else 0.
            per_set.append(dict(case=case.name,n=n,set=s,batch=batch,proofs=len(samples),query_seconds=timed,
                                one_time_seconds=one,total_seconds=timed+one,full_catalog_extrapolated=False))
        write_csv(a.out/(case.name+'_full_public_normalized.csv'),per_set)
        fullrows.append(dict(case=case.name,n=n,batch=batch,sets=sets,proofs=len(q['samples']),constraints=q['constraints'],
                             total_seconds=sum(x['total_seconds'] for x in per_set),
                             original_v1_total_seconds=summary['total_seconds'],
                             independently_verified_proofs=replay['verified'],peak_rss_bytes=summary['peak_rss_bytes']))
    (a.out/'completed_full_summary.json').write_text(json.dumps(fullrows,indent=2),encoding='utf-8')
    tex_table('completed_full.tex',
              'Completed public-center Full controls as of this draft. All catalog rows are actually proved for all ten sets under one query setup. Cost includes native answer construction; no catalog extrapolation or new reuse speedup is reported.',
              'tab:public-full-complete',['Catalog','Batch','Proofs','Constraints','Cost (s)','Peak RSS (GiB)'],
              [[x['case'].replace('_',r'\_'),str(x['batch']),str(x['proofs']),f"{x['constraints']:,}",f"{x['total_seconds']:,.1f}",f"{x['peak_rss_bytes']/2**30:.2f}"] for x in fullrows])
    manifest=dict(sources=records,source_files_hashed=True,native_not_proofs=True,
                  full_capture_all_hashes_valid=json.loads((a.full_capture/'capture.json').read_text())['all_transferred_hashes_valid'],
                  timing_policy='native prepare + compile + setup + native answer construction + witness + median proving; verification/warmup/fitting/I-O/public center preparation excluded',
                  normalization='First frozen Full worker omitted native_answer_seconds from its CSV total; new exports reconstruct this component from saved samples, without modifying the original output.')
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(out=str(a.out),full_cases=fullrows)))


if __name__=='__main__':main()
