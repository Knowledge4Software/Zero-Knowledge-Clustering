"""Update the private paper only after both matched proof campaigns finish.

Build/render output remains pending visual review; no publication or upload.
Unexpected intervening manuscript edits stop promotion instead of overwriting.
"""
from pathlib import Path
import argparse,json,os,re,shutil,subprocess,sys,time,zipfile
from commitjl.pilot_io import save,sha

REPO=Path(__file__).resolve().parents[2]
PARTS=['abstract','introduction','method','security','experiments','discussion','appendix']
TAG='neurocomputing_group_certificates_20261025'


def read(path):return json.loads(Path(path).read_text())


def atomic_copy(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_name(target.name+f'.{os.getpid()}.tmp');shutil.copy2(source,temp);os.replace(temp,target)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['root','candidate','replication-root','scale-root']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=a.resume);candidate=a.candidate.resolve()
    candidate_files=[candidate/'manuscript.tex']+[candidate/(name+'.tex') for name in PARTS]
    if a.resume:plan=read(root/'plan.json')
    else:
        plan=dict(canonical_before_sha256=sha(REPO/'manuscript.tex'),candidate_sha256={p.name:sha(p) for p in candidate_files},replication_root=str(a.replication_root.resolve()),scale_root=str(a.scale_root.resolve()),tag=TAG,permission='User authorized main experiments and manuscript update from their completed results. Private supervisor review only; no upload/publication.',visual_review='PDF generation/rendering is automated; all final page images still require inspection before final delivery.');save(root/'plan.json',plan)
    state=read(root/'queue.json') if a.resume else dict(complete=False,started_unix=time.time(),plan_sha256=sha(root/'plan.json'))
    state.update(pid=os.getpid(),status='waiting_for_proof_campaigns',child_pid=None,paused=False);save(root/'queue.json',state)
    try:
        while True:
            complete=True
            for queue in [a.replication_root/'queue.json',a.scale_root/'queue.json']:
                q=read(queue)
                if q.get('paused') or q.get('status')=='stopped':raise RuntimeError('Proof campaign paused/stopped; paper update requires explicit repair/resume')
                complete &= q.get('complete',False)
            if complete:break
            time.sleep(15)
        assert all(sha(path)==plan['candidate_sha256'][path.name] for path in candidate_files),'prepared draft changed; inspect before promotion'
        state.update(status='validating_and_exporting_results');save(root/'queue.json',state)
        from analysis.export_main_rank79_paper import export
        destination=REPO/'paper_results/main_revision_20261025';generated=destination/'generated'
        receipt=export(generated,a.replication_root.resolve(),a.scale_root.resolve(),preview=False);assert not receipt['preview'] and receipt['main_proof_rows']==11
        new_sha=sha(candidate/'manuscript.tex');current_sha=sha(REPO/'manuscript.tex')
        assert current_sha in [plan['canonical_before_sha256'],new_sha],'manuscript changed during runs; preserve edits and merge before promotion'
        backup=REPO/'paper_versions/20261025_before_main'
        if not backup.exists():
            backup.mkdir(parents=True);atomic_copy(REPO/'manuscript.tex',backup/'manuscript.tex');save(backup/'preservation_receipt.json',dict(canonical_sha256=current_sha,supervisor_original_sha256=sha(REPO/'paper_history/supervisor_draft.tex'),scope='Canonical master preserved before main-result update. All prior included sections and historical PDF releases remain unchanged.'))
        for name in PARTS:atomic_copy(candidate/(name+'.tex'),destination/(name+'.tex'))
        atomic_copy(candidate/'manuscript.tex',REPO/'manuscript.tex');save(root/'promotion_receipt.json',dict(complete=True,canonical_sha256=new_sha,export_receipt_sha256=sha(generated/'export_receipt.json'),supervisor_original_sha256=sha(REPO/'paper_history/supervisor_draft.tex'),private=True))
        build=REPO/'tmp/pdfs'/TAG;build.mkdir(parents=True,exist_ok=True);state.update(status='building_pdf');save(root/'queue.json',state)
        command=['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','-outdir='+str(build),'manuscript.tex']
        with (build/'console.log').open('ab') as log:
            child=subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));state['child_pid']=child.pid;save(root/'queue.json',state);code=child.wait()
        if code:raise RuntimeError('LaTeX build failed; inspect preserved console.log')
        log=(build/'manuscript.log').read_text(errors='replace');warnings=[line for line in log.splitlines() if any(v in line for v in ['Overfull','undefined references','undefined citations','Citation ','Float too large','multiply defined','LaTeX Error'])]
        if warnings:save(root/'layout_warnings.json',warnings);raise RuntimeError('Paper requires layout/reference fixes; inspect layout_warnings.json')
        from pypdf import PdfReader
        reader=PdfReader(build/'manuscript.pdf');texts=[p.extract_text() or '' for p in reader.pages];assert len(texts)>3 and all(len(s.strip())>150 for s in texts),'blank/near-empty page requires review'
        assert 'Author information pending supervisor confirmation' in texts[0]
        state.update(status='rendering_pages',child_pid=None);save(root/'queue.json',state)
        render=[shutil.which('pdftoppm') or 'pdftoppm','-scale-to','1700','-png',str(build/'manuscript.pdf'),str(build/'page')]
        with (build/'render.log').open('wb') as log:
            child=subprocess.Popen(render,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));state['child_pid']=child.pid;save(root/'queue.json',state);assert child.wait()==0
        assert len(list(build.glob('page-*.png')))==len(texts)
        # A preview pathname signals that page-by-page visual approval is still
        # pending; output/README does not relabel it as the reviewed final PDF.
        pdf=REPO/'output/pdf'/(TAG+'_review.pdf');atomic_copy(build/'manuscript.pdf',pdf)
        from analysis.build_three_papers import expand
        files=set();expand(REPO/'manuscript.tex',files);files.add(REPO/'references.bib');files.add(generated/'export_receipt.json')
        for folder in [a.replication_root.resolve(),a.scale_root.resolve()]:
            for name in ['summary.json','plan.json','queue.json']:
                if (folder/name).exists():files.add(folder/name)
        source={p.relative_to(REPO).as_posix():p.read_bytes() for p in sorted(files)}
        for name in ['elsarticle.cls','elsarticle-num.bst']:
            installed=Path(subprocess.check_output(['kpsewhich',name],text=True).strip());source[name]=installed.read_bytes()
        source['README_RELEASE.md']=b'# Private main-result paper\n\nCompile manuscript.tex with latexmk -pdf. Numerical exports are complete and hash checked; public-only proof replay and all fallback generation were completed before promotion. This is a paper/source package, not the full proof artifact. No publication or upload is authorized. Supervisor handles author/declaration fields. The rendered PDF still requires visual review before final delivery.\n'
        manifest={name:__import__('hashlib').sha256(data).hexdigest() for name,data in source.items()};source['source_manifest.json']=json.dumps(manifest,indent=2).encode();archive=REPO/'output'/(TAG+'_sources.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for name,data in source.items():z.writestr(name,data)
        with zipfile.ZipFile(archive) as z:assert all(__import__('hashlib').sha256(z.read(n)).hexdigest()==d for n,d in manifest.items())
        validation=dict(status='ready_for_visual_review',pdf=str(pdf.relative_to(REPO)),pdf_sha256=sha(pdf),source_archive=str(archive.relative_to(REPO)),source_sha256=sha(archive),pages=len(texts),page_images=str(build.relative_to(REPO)),latex_warnings=[],numerical_export='PASS',main_proof_rows=11,visual_review='PENDING: inspect every final page PNG before final delivery.',private=True)
        save(REPO/'output'/(TAG+'_validation.json'),validation)
        delivery=REPO/'output/README.md';previous=delivery.read_text(encoding='utf-8');marker='<!-- main-rank79-20261025 -->'
        if marker not in previous:
            note=marker+'\n## Main group-certificate revision, 25 October\n\nThe main matched proof campaigns and prospective fallback generation are complete. The canonical manuscript is updated from checked exports. The rendered PDF is **pending final visual review**.\n\n- [Paper for review](pdf/'+pdf.name+').\n- [Editable source package]('+archive.name+').\n- [Build/evidence receipt]('+TAG+'_validation.json).\n\nPrivate supervisor review; no publication or upload. Earlier paper releases below remain preserved.\n\n'
            delivery.write_text(note+previous,encoding='utf-8')
        state.update(complete=True,status='ready_for_visual_review',child_pid=None,finished_unix=time.time(),result=validation);save(root/'queue.json',state)
    except BaseException as error:state.update(status='stopped',error=str(error),child_pid=None);save(root/'queue.json',state);raise


if __name__=='__main__':main()
