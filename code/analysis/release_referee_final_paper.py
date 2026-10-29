"""Build the focused private paper; export only completed checked evidence.

Automatic post-campaign builds are candidates awaiting visual review. They
never overwrite supervisor-authored text or announce an uninspected PDF ready.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,zipfile
from campaigns.local_fast_tradeoff import save,sha
from analysis.export_referee_final_paper import export,DEST,REPO
from analysis.build_three_papers import expand

TAG='prototype_audit_referee_revision_20261026'


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,default=REPO/'code/validation/outputs/referee_final_20261025');p.add_argument('--tag',default=TAG);p.add_argument('--automatic',action='store_true');p.add_argument('--reuse-checked-tables',action='store_true',help='Presentation only: reuse complete checked evidence without recomputing native experiments');a=p.parse_args();campaign=a.campaign.resolve()
    pin=DEST.parent/'authoring_pin.json'
    authored=[REPO/'manuscript.tex',REPO/'references.bib']+list(DEST.parent.glob('*.tex'))
    if a.automatic:
        assert read(campaign/'summary.json')['complete'],'No post-campaign release from incomplete jobs'
        expected=read(pin)['authored_sha256'];assert all(sha(REPO/name)==h for name,h in expected.items()),'Authored paper changed during campaign; export needs manual rebase'
    if a.reuse_checked_tables:
        from analysis.reframe_checked_tables import reframe
        receipt=reframe(REPO,DEST,campaign)
    else:receipt=export(campaign,require_complete=a.automatic)
    # Bound wide comparative tables to the available journal text width.
    for path in DEST.glob('*.tex'):
        text=path.read_text()
        text=text.replace(r'\begin{table*}[t]',r'\begin{table*}[tbp]').replace(r'\begin{figure*}[t]',r'\begin{figure*}[tbp]')
        text=text.replace('group79 controls','group-79 controls').replace('prediction rank remains63','prediction rank remains 63').replace('selected79-dimensional','selected 79-dimensional').replace('frozen63-dimensional','frozen 63-dimensional').replace('completed128-row','completed 128-row')
        for old,new in [('rank63','rank 63'),('rank79','rank 79'),('cap16','cap 16'),('seed137','seed 137'),('seed149','seed 149'),('ages1/8','ages 1/8')]:text=text.replace(old,new)
        if r'\begin{table*}' in text and r'\begin{adjustbox}' not in text:
            text=text.replace(r'\begin{tabular}',r'\begin{adjustbox}{max width=.98\textwidth,center=\textwidth}'+'\n'+r'\begin{tabular}').replace(r'\end{tabular}',r'\end{tabular}'+'\n'+r'\end{adjustbox}')
        path.write_text(text,encoding='utf-8')
    if not a.automatic:save(pin,dict(authored_sha256={path.relative_to(REPO).as_posix():sha(path) for path in authored},scope='Guard against overwriting supervisor edits during automatic completed-evidence export.'))
    suffix='_completed_candidate' if a.automatic else ''
    tag=a.tag+suffix;build=REPO/'tmp/pdfs'/tag;build.mkdir(parents=True,exist_ok=True)
    with (build/'console.log').open('wb') as log:
        result=subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','-outdir='+str(build),'manuscript.tex'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0,'LaTeX failed; inspect '+str(build/'console.log')
    log=(build/'manuscript.log').read_text(errors='replace');warnings=[line for line in log.splitlines() if any(token in line for token in ['Overfull','undefined references','undefined citations','Float too large','multiply defined','LaTeX Error'])]
    assert not warnings,warnings
    from pypdf import PdfReader
    reader=PdfReader(build/'manuscript.pdf');assert not reader.is_encrypted,'Deliver an unprotected PDF'
    pages=reader.pages;text='\n'.join(page.extract_text() or '' for page in pages)
    focused=(DEST.parent/'evaluation_two_reviews.tex').exists() and 'evaluation_two_reviews' in (DEST.parent/'experiments.tex').read_text()
    required=['moments','refinement','HAR','2.388','BN254','minibatch'] if focused else ['moments','refinement','196.703','61.637','40.174','24.771','39.457','25/25','15/16']
    for token in required:assert token in text,'missing '+token
    assert 'Compact head 63' not in text and 'JL ablations' not in text
    assert sha(REPO/'paper_history/supervisor_draft.tex')=='4f06491ac2e89b747083d42f9983289b278de6835dc86d2bcac7fd2294f93a69'
    pdf=REPO/'output/pdf'/(tag+'.pdf');shutil.copy2(build/'manuscript.pdf',pdf)
    render=build/'renders'/sha(pdf)[:12];render.mkdir(parents=True,exist_ok=True)
    with (build/'render.log').open('wb') as log:
        result=subprocess.run(['pdftoppm','-scale-to','1700','-png',str(pdf),str(render/'page')],stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0 and len(list(render.glob('page-*.png')))==len(pages)
    files=set();expand(REPO/'manuscript.tex',files);files.add(REPO/'references.bib');files.update(DEST.glob('*.json'));files.add(pin)
    optimized=DEST/'optimized_evidence_receipt.json'
    if optimized.exists():
        files.update(REPO/name for name in read(optimized)['source_sha256'])
        files.add(REPO/'code/analysis/update_optimized_evidence.py')
        files.add(REPO/'code/analysis/benchmark_final_kernels.py')
        for name in ['backend_runtime_amendment.json','extended_stress_amendment.json']:
            path=REPO/'code/validation/outputs/referee_optimized_20261026'/name
            if path.exists():files.add(path)
        for receipt_name in ['optimized_timing_receipt.json','optimized_exact_baseline_receipt.json']:
            path=DEST/receipt_name
            if path.exists():
                for value in read(path).get('records',[]):
                    if 'receipt' in value:files.add(REPO/value['receipt'])
    reviews=DEST/'two_review_evidence_receipt.json'
    if reviews.exists():
        files.update(REPO/name for name in read(reviews)['source_sha256'])
        for name in ['analysis/export_two_review_evidence.py','campaigns/referee_two_reviews_local.py','campaigns/referee_sleep_guardian.py','experiments/referee_learning_workflows.py','data/referee_revision_data.py']:
            files.add(REPO/'code'/name)
        for record in read(reviews)['completed_plonk_runs']+read(reviews).get('completed_groth_backend_controls',[]):
            evidence=REPO/record['receipt'];files.add(evidence)
            if (evidence.parent/'public_verify.json').exists():files.add(evidence.parent/'public_verify.json')
        for record in read(reviews).get('new_proof_pairs',[]):files.add(REPO/record['receipt'])
        for name in ['experiments/neural_activity_prototypes.py','campaigns/referee_review_followups.py','zk/cmd/full-plonk-control/main.go','zk/go.mod','zk/go.sum']:
            files.add(REPO/'code'/name)
    for name in ['release_referee_final_paper.py','export_referee_final_paper.py','refinement_paper_evidence.py','check_moments_only_necessity.py','reframe_checked_tables.py']:files.add(REPO/'code/analysis'/name)
    for name in ['refinement_extension.json','refinement_execution_amendment.json','memory_policy_amendment.json','lazy_import_amendment.json']:
        if (campaign/name).exists():files.add(campaign/name)
    for folder in ['refinement_confirmation_20261025','selective_switch_refinement_20261025_v2']:
        for name in ['protocol.json','summary.json','records.json']:
            files.add(REPO/'code/validation/outputs'/folder/name)
    files.add(campaign/'plan.json');files.add(REPO/'code/validation/outputs/referee_necessity_check_20261025/summary.json')
    extra_replay=campaign/'public_replay_round3_20261026/status.json'
    if extra_replay.exists():files.add(extra_replay)
    data={path.relative_to(REPO).as_posix():path.read_bytes() for path in sorted(files)}
    for name in ['elsarticle.cls','elsarticle-num.bst']:
        path=Path(subprocess.check_output(['kpsewhich',name],text=True).strip());data[name]=path.read_bytes()
    data['README_RELEASE.md']=b'# Focused referee revision, private review\n\nCompile manuscript.tex with latexmk -pdf. Includes one representation-agnostic certificate with centroid-informed/PCA bases, reusable exact Full, authenticated moments and separate known-family affine comparisons. Native quality and measured proof timings are separate. Unfinished comparisons are excluded; no public artifact is claimed. This package contains editable manuscript sources and derived evidence receipts, not the full private raw proof repository. New protocol definitions are conditional specifications, not a composable-security audit.\n'
    manifest={name:hashlib.sha256(value).hexdigest() for name,value in data.items()};data['source_manifest.json']=json.dumps(manifest,indent=2).encode();archive=REPO/'output'/(tag+'_sources.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,value in data.items():z.writestr(name,value)
    with zipfile.ZipFile(archive) as z:assert all(hashlib.sha256(z.read(name)).hexdigest()==h for name,h in manifest.items())
    validation=dict(status='awaiting_visual_review',pdf=pdf.relative_to(REPO).as_posix(),pdf_sha256=sha(pdf),source_zip=archive.relative_to(REPO).as_posix(),source_zip_sha256=sha(archive),pages=len(pages),render_directory=render.relative_to(REPO).as_posix(),latex_warnings=[],no_pending_results_reported=True,visual_review='PENDING',complete_matched_cohorts=receipt['complete_matched_cohorts'],all_matched_complete=receipt['all_matched_complete'],supervisor_original_preserved=True,authored_sources_preserved=True,automatic_candidate=a.automatic)
    if optimized.exists():
        study=REPO/'code/validation/outputs/referee_optimized_20261026'
        timing=read(DEST/'optimized_timing_receipt.json')
        extended=read(study/'extended_status.json') if (study/'extended_status.json').exists() else {}
        validation.update(native_study_complete=True,native_representation_fits=56,native_records=280,
            new_matched_repetitions_complete=timing['completed_repetitions'],all_requested_extended_proofs_complete=extended.get('complete',False),
            large_scale_proof_timings_pending=not extended.get('complete',False),unresolved_scientific_issues=True,submission_ready=False)
        scope=read(optimized).get('user_execution_scope')
        if scope:validation.update(maximum_proof_catalog_records=scope['maximum_proof_catalog_records'],
            large_scale_proof_timings_pending=False,large_scale_proof_timings_deferred=True)
    if reviews.exists():
        new=read(reviews)
        validation.update(new_learning_quality_rows=len(new['qualities'])-4,
            completed_new_backend_executions=len(new['completed_plonk_runs']),
            completed_new_learning_proof_pairs=len(new.get('new_proof_pairs',[])),
            new_evidence_receipt_sha256=sha(reviews),new_campaign_status='See output/two_review_campaign_status_20261027.json; unfinished experiments are excluded.')
    save(REPO/'output'/(tag+'_validation.json'),validation);print(json.dumps(validation,indent=2))


if __name__=='__main__':main()
