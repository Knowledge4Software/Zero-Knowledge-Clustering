"""Integrate the completed methodology and a CCD-structured pilot evaluation.

No experiments are launched. Existing early prose, bibliography, original
methodology and prior results are preserved. New tables read checked receipts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess

REPO=Path(__file__).resolve().parents[2]
DEST=REPO/'paper_results/ccd_pilot_20261029'
HISTORY=REPO/'paper_history/ccd_pilot_before_20261029'
BUILD=REPO/'tmp/pdfs/ccd_pilot_20261029'
TAG='prototype_audit_ccd_pilot_20261029'


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):return json.loads((REPO/p).read_text(encoding='utf-8'))
def write(p,text):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8',newline='\n')


def prepare():
    HISTORY.mkdir(parents=True,exist_ok=True)
    original=(REPO/'manuscript.tex').read_bytes()
    if (HISTORY/'manuscript.tex').exists():
        assert (HISTORY/'manuscript.tex').read_bytes()==original,'Do not overwrite an edited root manuscript'
    else:(HISTORY/'manuscript.tex').write_bytes(original)
    protected=[REPO/'references.bib',REPO/'Sparse_JL_for_ZK_Clustering.zip',REPO/'Spectral_CCD.zip']
    protected+=list((REPO/'paper_results/referee_final_20261025').glob('*.tex'))
    protected+=list((REPO/'paper_results/supervisor_methodology_20261027').glob('*.*'))
    protected=[p for p in protected if p.is_file()]
    prefix=original.split(b'\\input{paper_results/referee_final_20261025/method.tex}')[0]
    record=dict(protected_sha256={str(p.relative_to(REPO)):sha(p) for p in protected},
                original_root_sha256=sha(REPO/'manuscript.tex'),
                original_root_prefix_hex=prefix.hex(),source_zip_preserved=True)
    write(HISTORY/'preservation.json',json.dumps(record,indent=2))
    DEST.mkdir(exist_ok=True)
    for name in ['methodology.tex','methodology_appendix.tex']:
        shutil.copy2(REPO/'paper_results/supervisor_methodology_20261027'/name,DEST/name)
    root=original.decode('utf-8')
    newline='\r\n' if '\r\n' in root else '\n'
    old='\\input{paper_results/referee_final_20261025/method.tex}\n\\input{paper_results/referee_final_20261025/security.tex}\n\\input{paper_results/referee_final_20261025/experiments.tex}\n\\input{paper_results/referee_final_20261025/discussion.tex}'
    new='''% Completed methodology text is preserved verbatim in the integrated copy.
\\makeatletter
\\def\\input@path{{paper_results/ccd_pilot_20261029/}}
\\makeatother
\\input{paper_results/ccd_pilot_20261029/methodology.tex}
\\input{paper_results/ccd_pilot_20261029/methodology_figures.tex}
\\input{paper_results/ccd_pilot_20261029/experimental_design.tex}
\\input{paper_results/ccd_pilot_20261029/experimental_results.tex}
\\input{paper_results/ccd_pilot_20261029/discussion.tex}
\\input{paper_results/ccd_pilot_20261029/threats_to_validity.tex}
\\input{paper_results/ccd_pilot_20261029/conclusions.tex}'''
    old=old.replace('\n',newline);new=new.replace('\n',newline)
    assert old in root
    root=root.replace(old,new,1).replace('\\input{paper_results/referee_final_20261025/appendix.tex}',
        '\\input{paper_results/ccd_pilot_20261029/methodology_appendix.tex}'+newline+'\\input{paper_results/ccd_pilot_20261029/evidence_appendix.tex}')
    root=root.replace('\\section*{Declarations}','\\FloatBarrier'+newline+'\\section*{Declarations}',1)
    (REPO/'manuscript.tex').write_bytes(root.encode('utf-8'))
    assert (REPO/'manuscript.tex').read_bytes().startswith(prefix)
    print(json.dumps(dict(prepared=True,preserved_methodology=True,destination=str(DEST))))


def table(name,caption,label,columns,header,rows,wide=True):
    kind='table*' if wide else 'table'
    content='\\begin{'+kind+'}[tbp]\n\\centering\n\\small\n'
    content+='\\caption{'+caption+'}\n\\label{'+label+'}\n'
    content+='\\begin{adjustbox}{max width='+('\\textwidth' if wide else '\\columnwidth')+'}\n'
    if wide:
        content+='\\begin{tabular*}{.98\\textwidth}{@{\\extracolsep{\\fill}}'+columns+'@{}}\n'
    else:content+='\\begin{tabular}{'+columns+'}\n'
    content+='\\toprule\n'+header+'\\\\\n\\midrule\n'
    content+='\n'.join(' & '.join(map(str,row))+'\\\\' for row in rows)
    content+='\n\\bottomrule\n\\end{'+('tabular*' if wide else 'tabular')+'}\n\\end{adjustbox}\n\\end{'+kind+'}\n'
    write(DEST/'generated'/name,content)


def evidence():
    sources={}
    def take(path):sources[path]=sha(REPO/path);return read(path)
    four=take('output/laptop_four_hour_research_20261028.json')
    follow=take('output/request_crossover_followup_20261028.json')
    aware=take('output/cost_aware_20261029.json')
    final=take('output/cost_aware_final_20261029.json')
    split=take('output/request_cost_split_20261028.json')
    old=take('paper_results/referee_final_20261025/generated/optimized_timing_receipt.json')
    necessity=take('code/validation/outputs/referee_necessity_check_20261025/summary.json')
    native=take('code/validation/outputs/laptop_four_hour_research_20261028/native_ablations.json')
    take('code/validation/outputs/cost_aware_20261029/smoke_v2/checks.json')
    take('code/validation/outputs/cost_aware_v2_20261029/final_evidence_checks.json')
    take('code/validation/outputs/cost_aware_20261029/final_evidence_checks.json')
    assert follow['complete'] and aware['complete'] and final['complete'] and native['new_proofs']==0
    cold=[]
    for case,experiment,q,title in [('fashion_n32','basic_cold',8,'Fashion'),('text_n32','basic_cold',8,'Text'),('fashion_n128','scaled_cold',4,'Fashion')]:
        records=[r for r in four['measurements'] if r['case']==case and r['experiment']==experiment]
        by={r['method']:r for r in records}
        names=['Exact Full / Groth16','ZKML primitive adapter / Halo2-KZG','Centroid certificate','PCA certificate']
        vals=[f"{by[m]['seconds']/60:.3f}" if m in by else '--' for m in names]
        ours=by['Centroid certificate']
        cold.append([title,ours['n'],q,*vals,f"{ours['coverage']}/{q}",ours['fallback']])
    by={r['method']:r for r in follow['records']}
    vals=[f"{by[m]['seconds']/60:.3f}" for m in ['Exact Full / Groth16','ZKML primitive adapter / Halo2-KZG','Centroid certificate','PCA certificate']]
    ours=by['Centroid certificate'];cold.append(['Text',32,16,*vals,f"{ours['coverage']}/16",ours['fallback']])
    table('cold_pilots.tex','Completed exploratory cold campaign comparisons. Times are minutes and include the preparation, generation, replay and paid fallback retained by each campaign. Q8 totals retain recorded fitting; the Q16 follow-up supplies frozen fitted inputs. Coverage and FB concern the centroid certificate before and after fallback. The missing PCA N128 entry was interrupted by a host-memory guard.','tab:ccd-cold','lrrrrrrrr','Data & $n$ & $Q$ & Full & ZKML-ad. & Centroid & PCA & Cov. & FB',cold)
    warm=[]
    f=next(c for c in aware['cases'] if c['name']=='fashion_n32')
    t=next(c for c in aware['cases'] if c['name']=='text_n32')
    w=four['warm64']
    warm.append(['Fashion',8,'Cost-aware',f"{f['routed_seconds']/60:.3f}",'Full',f"{f['full_seconds']/60:.3f}",f"{f['full_over_routed']:.3f}"])
    warm.append(['Fashion',64,'Refine-first',f"{w['measured_warm_certificate_seconds']/60:.3f}",'ZKML-ad.',f"{w['measured_warm_exact_seconds']/60:.3f}",f"{w['exact_over_certificate']:.3f}"])
    warm.append(['Text',16,'Mixed route',f"{t['routed_seconds']/60:.3f}",'Full',f"{t['full_seconds']/60:.3f}",f"{t['full_over_routed']:.3f}"])
    warm.append(['Text',16,'Final all-Full',f"{final['final_text_warm_seconds']/60:.3f}",'Unpaired','--','--'])
    table('warm_pilots.tex','Measured warm proof/query totals for $n=32$; times are minutes. Each paired row reuses setup and applicable old evidence for both arms and generates new request proofs. Ratios compare the reference with the certificate service within that row. The final text rerun has no new paired reference. Policies and request counts differ across rows.','tab:ccd-warm','lrlrlrr','Data & $Q$ & Policy & Service & Reference & Ref. & Ratio',warm)
    prep=[]
    for data,execution,q in [('fashion_n32','previous8',8),('text_n32','new16',16)]:
        rs=[r for r in split['records'] if r['case']==data and r['execution']==execution]
        for method in ['Full-Groth16','ZKML-adapter','Centroid','PCA-certificate']:
            r=next(x for x in rs if x['method']==method)
            prep.append(['Fashion' if data.startswith('fashion') else 'Text',q,{'Full-Groth16':'Full','ZKML-adapter':'ZKML-ad.','Centroid':'Centroid','PCA-certificate':'PCA'}[method],
                f"{r['native_owner_prepare_seconds']:.2f}",f"{r['summary_authentication_seconds']:.2f}",f"{r['base_key_compile_setup_seconds']:.2f}",f"{r['identified_reusable_seconds']:.2f}",f"{r['response_residual_mean_seconds']:.2f}"])
    table('cost_components.tex',r'Component accounting of the earlier refine-first cold campaigns, in seconds. Native preparation, row authentication and base keys are separately observed stages. $S$ also includes recorded fitting, lazy correction setup and fallback/reference preparation where applicable. $\bar R$ is the remaining phase total divided by $Q$, not separately measured hot-RPC latency. Enclosing/export gaps are retained in the source ledger. These are not cold timings of the final router.','tab:ccd-components','lrlrrrrr',r'Data & $Q$ & Method & Native & Auth. & Keys & $S$ & $\bar R$',prep)
    ab=[]
    for n in [32,128]:
        for rank in [4,39,63,79]:
            pair=[]
            for method in ['ordered_head','pca']:
                rows=[r for r in native['records'] if r['dataset']=='fashion' and r['n']==n and r['certificate_rank']==rank and r['cap']==16 and r['method']==method]
                assert len(rows)==8
                pair.extend([f"{sum(r['decision']!='inconclusive' for r in rows)}/8",sum(r['exact_rows'] for r in rows)])
            ab.append([n,rank,*pair])
    table('native_ranks.tex','Native-only Fashion ablation; prediction rank remains 63 and the correction cap is 16. Cov. is conclusive update coverage and Rows is the total corrected-row count across eight requests. No proving time is inferred.','tab:ccd-ranks','rrrrrr','$n$ & $s$ & Cent. cov. & Rows & PCA cov. & Rows',ab,wide=False)
    archive=[]
    for method in ['Centroid','Moments-only','PCA']:
        r=next(r for r in old['records'] if r['n']==1024 and r['method']==method)
        archive.append([method,f"{r['seconds']/60:.3f}",f"{r['full_seconds']/60:.3f}",f"{r['ratio']:.3f}",f"{r['decisions']}/{r['requests']}",r['fallback']])
    table('archived_control.tex','Archived completed Fashion control, $n=1024$, $k=5$, seed 173 and reference step 8. Frozen models are supplied and eight workers are used. Times are minutes, including preparation, proof work, replay and paid fallback. This is a separate workload from the new seed 521 pilots.','tab:ccd-archive','lrrrrr','Method & Min. & Full min. & Ratio & Cov. & FB',archive,wide=False)
    source=necessity['cases'][0]
    assert source['updates']==19 and source['current_conclusive']==19 and source['best_groupwise_improvements']==2
    write(DEST/'evidence_manifest.json',json.dumps(dict(complete=True,sources_sha256=sources,
          new_proofs_run=False,server_results_available=False,generated_tables=6,
          methodology_text_preserved=True,results_status='completed local evidence; preliminary before new server campaign'),indent=2))
    print(json.dumps(dict(tables=6,sources=len(sources),new_proofs=False)))


def build():
    BUILD.mkdir(parents=True,exist_ok=True)
    with (BUILD/'console.log').open('wb') as log:
        r=subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','-outdir='+str(BUILD),'manuscript.tex'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps(dict(returncode=r.returncode,log=str(BUILD/'console.log'))))
    if r.returncode:raise SystemExit(r.returncode)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','evidence','build']);a=p.parse_args()
    globals()[a.phase]()
