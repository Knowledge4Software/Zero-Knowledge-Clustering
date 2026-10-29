"""Publish a private current-evidence paper locally while experiments are paused.

No proof generation or queue resume. All reported proof cases must be complete.
"""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,zipfile
from commitjl.pilot_io import save,sha
from analysis.export_main_rank79_paper import export,table,quality_for_group
from campaigns.main_rank79 import verify_group
from analysis.audit_local_fast_tradeoff import audit_case
from analysis.build_three_papers import expand

REPO=Path(__file__).resolve().parents[2]
TAG='neurocomputing_current_results_20261025'
PARTS=['abstract','introduction','method','security','experiments','discussion','appendix']


def read(path):return json.loads(Path(path).read_text())


def write(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.release.tmp');temp.write_text(text,encoding='utf-8');os.replace(temp,path)


def main():
    registry=read(REPO/'code/validation/outputs/research_jobs_20261024.json');assert registry['paused_by_user'] and not registry['active_processes']
    root=REPO/'code/validation/outputs/rank79_independent_replication_20261025/fashion_mnist_k3_seed101_age8/head79';result=verify_group(root);quality=quality_for_group(root)
    parent=REPO/'paper_results/review_controls/rank79_independent_replication_20261025/proof_inputs/fashion_mnist_k3_seed101_age8_ordered_head';full,_,_=audit_case(parent,'full');assert full['active_phase_wall_seconds']==result['full_reference_active_seconds']
    assert result['proposal_updates']==result['conclusive_without_fallback']==2 and quality['distinct_regressions']==quality['regressions_certified_without_fallback']==2
    destination=REPO/'paper_results/main_revision_20261025';generated=destination/'generated';candidate=REPO/'paper_versions/20261025_main_candidate'
    backup=REPO/'paper_versions/20261025_before_current_results'
    if not backup.exists():
        backup.mkdir();shutil.copy2(REPO/'manuscript.tex',backup/'manuscript.tex');save(backup/'preservation_receipt.json',dict(canonical_sha256=sha(REPO/'manuscript.tex'),supervisor_sha256=sha(REPO/'paper_history/supervisor_draft.tex'),scope='Previous master preserved. Its included source sections and earlier PDF releases are unchanged.'))
    for name in PARTS:shutil.copy2(candidate/(name+'.tex'),destination/(name+'.tex'))
    receipt=export(generated,REPO/'code/validation/outputs/rank79_independent_replication_20261025',REPO/'code/validation/outputs/main_rank79_20261025',preview=True)
    macros=dict(FreshProofMinutes=f"{result['active_phase_seconds_including_fitting_and_fallback']/60:.3f}",FreshFullMinutes=f"{full['active_phase_wall_seconds']/60:.3f}",FreshProofSpeedup=f"{result['full_over_group_active_phase']:.3f}",FreshProofCount=result['group_public_proofs'],FreshProofAgreement=f"{quality['mean_agreement']:.3f}")
    metrics=(generated/'metrics.tex').read_text();metrics+='\n% Completed fresh replication; expanded main queues remain incomplete.\n'+'\n'.join('\\providecommand{\\'+k+'}{'+str(v)+'}' for k,v in macros.items())+'\n';write(generated/'metrics.tex',metrics)
    rows=[f"Reusable exact Full & {full['active_phase_wall_seconds']/60:.3f} & 1.000 & {full['public_proofs']}",f"Compact auxiliary 79 & {result['active_phase_seconds_including_fitting_and_fallback']/60:.3f} & {result['full_over_group_active_phase']:.3f} & {result['group_public_proofs']}"]
    caption='Completed fresh-fit Fashion-MNIST mature-model replication, $N=128$, $k=3$, seed 101. Both distinct natural updates worsen held-out cost and are certified without fallback. Times include cold setup and fresh public-only replay; this is one fit and one timing repetition.'
    write(generated/'completed_replication.tex',table(caption,'tab:group-completed-replication','lrrr',r'Method & Minutes & Full/method & Proofs',rows))
    abstract=(destination/'abstract.tex').read_text();marker='The lower-rank profile takes'
    addition='A completed fresh-fit Fashion-MNIST replication certifies two distinct\n naturally worsening updates on 128 records, with \\FreshProofCount{} freshly\n verified proofs, zero-width cost-change intervals and no fallback. It takes\n \\FreshProofMinutes{} active minutes versus \\FreshFullMinutes{} for Full,\n a \\FreshProofSpeedup-fold ratio. This small same-fit cohort does not\n establish broad regression sensitivity.\n'
    assert marker in abstract;write(destination/'abstract.tex',abstract.replace(marker,addition+marker))
    experiments=(destination/'experiments.tex').read_text();marker=r'\subsection{JL ablations and historical negative evidence}'
    addition=r'''\subsection{Completed fresh-fit proof replication}
The predeclared Fashion-MNIST mature workload ($k=3$, seed 101, deployed
iteration 8) completed both the public-center Full reference and the
auxiliary-79 group deployment. The four requested training-only Lloyd
snapshots contain two distinct nonidentity candidate sets; repeated final
snapshots are cached before evaluation. On this 128-row catalog both distinct
updates naturally worsen exact held-out cost. These outcomes differ from
the 1,024-row catalog and are reported separately.

\input{paper_results/main_revision_20261025/generated/completed_replication.tex}

All \FreshProofCount{} group proofs and 48 Full proofs passed fresh public-only
verification. All 154 group public-manifest file hashes match. Both group
intervals are zero-width and equal the independently recomputed Full changes.
Their exactness arises because the checked switching corrections are zero:
neither update changes a winning membership on this catalog. This is a
useful mature-model regime, not a guarantee that switching updates have zero
uncertainty. Both natural regressions are certified with no exact fallback.
The frozen predictor's agreement and its separately computed assignment
certificate fraction are both 100\% for these two sets; these label metrics
are native checks, not additional outputs of the group proof.

The active Full/method ratio is \FreshProofSpeedup{}, compared with
\GroupSeventyNineSpeedup{} on the original 19-improvement workload. This
replication has only two distinct updates and amortizes preprocessing less;
model, dataset and host load also differ, so the ratio difference cannot be
attributed to amortization alone. PCA and rank-63 proof costs on this fit are
not yet complete. The larger group-certificate proof controls remain outside
the evidence cutoff. No completed nonzero-fallback-generation timing is
claimed for the selected group profile.

'''
    assert marker in experiments;write(destination/'experiments.tex',experiments.replace(marker,addition+marker))
    introduction=(destination/'introduction.tex').read_text();marker='A lower-rank profile is\nfaster but gives wider intervals.'
    replacement='A lower-rank profile is\nfaster but gives wider intervals. A separate completed mature Fashion-MNIST\nproof workload certifies both naturally worsening updates without fallback\nand takes \\FreshProofMinutes{} minutes versus \\FreshFullMinutes{} for Full\n(\\FreshProofSpeedup-fold ratio). These two cases share one fitted model;\ntheir zero-width intervals arise from no switching and do not eliminate\ninconclusive outcomes elsewhere.'
    assert marker in introduction;write(destination/'introduction.tex',introduction.replace(marker,replacement))
    discussion=(destination/'discussion.tex').read_text();discussion=discussion.replace('This completed control does not establish the cost of nonzero fallback or\nperformance across new fits.','The completed fresh Fashion-MNIST control adds one fitted model and two\nnaturally worsening updates with a \\FreshProofSpeedup-fold ratio and no\nfallback. It still does not establish the cost of nonzero fallback or a\nperformance distribution across fits.')
    discussion=discussion.replace('An exact known-family affine service is both\nfaster while returning exact costs and assignments on the measured closed-bank workload, and should be','An exact known-family affine service is\nfaster while returning exact costs and assignments on the measured closed-bank\nworkload, and should be');write(destination/'discussion.tex',discussion)
    master=(candidate/'manuscript.tex').read_text().replace('% Candidate focused revision; compile from project root, not this directory.','% Current-results revision, 25 October 2026; completed evidence only.').replace('% Completed controls only; new matched proof campaign remains pending.','% Experiments paused by the user; uncompleted main cases excluded.');write(REPO/'manuscript.tex',master)
    sources=receipt['sources'];sources.update({str(p.relative_to(REPO)):sha(p) for p in [root/'checked_comparison.json',root/'native_checks.json',root/'public/public_manifest.json',root/'public/group_verify.json',parent/'full/public_bundle/verify.json']})
    current=dict(complete=True,expanded_queues_complete=False,experiments_paused=True,scope='Current completed evidence cutoff; no pending-case timing inferred.',completed_fresh_replication=dict(result=result,quality=quality,full=full),sources=sources,generated_sha256={p.name:sha(p) for p in generated.glob('*.tex')},supervisor_sha256=sha(REPO/'paper_history/supervisor_draft.tex'));save(generated/'current_release_receipt.json',current)
    build=REPO/'tmp/pdfs'/TAG;build.mkdir(parents=True,exist_ok=True)
    with (build/'console.log').open('wb') as log:r=subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','-outdir='+str(build),'manuscript.tex'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    assert r.returncode==0,'LaTeX failed; inspect console.log'
    log=(build/'manuscript.log').read_text(errors='replace');warnings=[s for s in log.splitlines() if any(v in s for v in ['Overfull','undefined references','undefined citations','Float too large','multiply defined','LaTeX Error'])];assert not warnings,warnings
    from pypdf import PdfReader
    pages=PdfReader(build/'manuscript.pdf').pages;text='\n'.join(p.extract_text() or '' for p in pages)
    for token in ['1.593','24.771','39.457','25/25','24/25','144/148','146']:assert token in text,'missing result '+token
    assert sha(REPO/'paper_history/supervisor_draft.tex')=='4f06491ac2e89b747083d42f9983289b278de6835dc86d2bcac7fd2294f93a69'
    pdf=REPO/'output/pdf'/(TAG+'.pdf');shutil.copy2(build/'manuscript.pdf',pdf)
    with (build/'render.log').open('wb') as log:r=subprocess.run(['pdftoppm','-scale-to','1700','-png',str(pdf),str(build/'page')],stdout=log,stderr=subprocess.STDOUT)
    assert r.returncode==0 and len(list(build.glob('page-*.png')))==len(pages)
    files=set();expand(REPO/'manuscript.tex',files);files.add(REPO/'references.bib');files.add(generated/'current_release_receipt.json');files.update(generated.glob('*.json'));files.add(REPO/'code/analysis/release_current_rank79_paper.py');files.add(REPO/'code/analysis/export_main_rank79_paper.py')
    for folder in ['paper_results/review_controls/completed_fashion_replication_20261025','paper_results/review_controls/rank79_independent_replication_20261025','paper_results/review_controls/residual_certificate_rank_confirmation_20261025']:
        for name in ['summary.json','distinct_summary.json','fit_records.json','results.csv']:
            p=REPO/folder/name
            if p.exists():files.add(p)
    data={p.relative_to(REPO).as_posix():p.read_bytes() for p in sorted(files)}
    for name in ['elsarticle.cls','elsarticle-num.bst']:
        p=Path(subprocess.check_output(['kpsewhich',name],text=True).strip());data[name]=p.read_bytes()
    data['README_RELEASE.md']=b'# Current completed-evidence manuscript, 25 October 2026\n\nCompile manuscript.tex with latexmk -pdf. Includes completed original MNIST controls, first completed fresh Fashion-MNIST group79/Full proof replication and distinct-count native studies. Other proof jobs are user-paused, and main N256/N1024 group runs never started. No unfinished timing is reported. This paper/source package is not the full raw proof artifact. Supervisor handles final author/declaration fields. Private review only: no upload/publication.\n'
    manifest={n:hashlib.sha256(v).hexdigest() for n,v in data.items()};data['source_manifest.json']=json.dumps(manifest,indent=2).encode();archive=REPO/'output'/(TAG+'_sources.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for n,v in data.items():z.writestr(n,v)
    with zipfile.ZipFile(archive) as z:assert all(hashlib.sha256(z.read(n)).hexdigest()==v for n,v in manifest.items())
    validation=dict(status='awaiting_visual_review',pdf=str(pdf.relative_to(REPO)),pdf_sha256=sha(pdf),source_zip=str(archive.relative_to(REPO)),source_zip_sha256=sha(archive),pages=len(pages),build_returncode=0,latex_warnings=[],source_files=len(data),experiments_paused=True,no_pending_results_reported=True,visual_review='PENDING',supervisor_original_preserved=True);save(REPO/'output'/(TAG+'_validation.json'),validation)
    print(json.dumps(validation,indent=2))


if __name__=='__main__':main()
