"""Read-only cleanup plan: retired outputs and rebuildable inactive caches.

Deletion is performed separately by the native PowerShell cleanup script.
Current paper evidence, all public verification artifacts in validation, and
the active/future proof cache directories are protected explicitly.
"""
from pathlib import Path
import hashlib,json,os,time
from collections import defaultdict
import psutil

ROOT=Path(__file__).resolve().parents[2]
LIVE=ROOT/'code/validation/outputs/referee_two_reviews_20261027'
AFFINE=ROOT/'code/validation/outputs/referee_optimized_20261026/extended/fashion_mnist_k5_seed173_age8_n1024/repetition_01/online_affine'
OUT=ROOT/'output/project_cleanup_20261027_plan.json'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    pins={}
    def walk(value):
        if isinstance(value,dict):
            for name,v in value.items():
                if isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v) and '/' in name:
                    path=(ROOT/name).resolve()
                    if path.is_relative_to(ROOT) and path.is_file():pins[path]=v
                walk(v)
        elif isinstance(value,list):
            for v in value:walk(v)
    for receipt in (ROOT/'paper_results/referee_final_20261025/generated').glob('*.json'):
        walk(json.loads(receipt.read_text()))
    for name in ['paper_results/referee_final_20261025/authoring_pin.json',
                 'paper_results/supervisor_methodology_20261027/methodology_checks.json']:
        value=json.loads((ROOT/name).read_text());walk(value)
        for field in ['authored_sha256','code_source_sha256']:
            for path,digest in value.get(field,{}).items():pins[(ROOT/path).resolve()]=digest
    for name in ['manuscript.tex','references.bib','paper_history/supervisor_draft.tex',
                 'Sparse_JL_for_ZK_Clustering.zip','Spectral_CCD.zip',
                 'output/pdf/prototype_audit_two_reviews_20261027.pdf','output/pdf/prototype_update_methodology_20261027.pdf',
                 'output/prototype_audit_two_reviews_20261027_sources.zip','output/supervisor_methodology_20261027.zip']:
        p=ROOT/name;pins[p]=sha(p)
    for p in (ROOT/'paper_results/supervisor_methodology_20261027').glob('*.tex'):pins[p]=sha(p)
    # Validate immutable evidence before constructing any deletion proposal.
    for p,digest in pins.items():
        if sha(p)!=digest:raise RuntimeError('Evidence pin changed before cleanup: '+str(p))
    protected=[ROOT/'.git',LIVE,AFFINE,ROOT/'code/.cache/commitjl_data',
               ROOT/'tmp/pdfs/prototype_audit_two_reviews_20261027',ROOT/'tmp/pdfs/supervisor_methodology_20261027',
               ROOT/'paper_results/referee_final_20261025',ROOT/'paper_results/supervisor_methodology_20261027']
    candidates={}
    def add(path,reason):
        path=path.resolve()
        if not path.exists():return
        if path==ROOT or not path.is_relative_to(ROOT):raise RuntimeError('Target escapes workspace')
        if any(path==p or path.is_relative_to(p) or p.is_relative_to(path) for p in protected):return
        if any(path==p or p.is_relative_to(path) for p in pins):raise RuntimeError('Target contains pinned evidence: '+str(path))
        if any(parent.is_junction() or parent.is_symlink() for parent in [path]+list(path.parents) if parent.is_relative_to(ROOT)):
            raise RuntimeError('Refuse linked cleanup target: '+str(path))
        candidates[path]=reason
    for name in ['paper_versions','paper_assets','figures','paper_results/main_revision_20261025',
                 'paper_results/reuse_revision','paper_results/bound_revision','paper_results/figures']:
        add(ROOT/name,'Superseded manuscript snapshots, unused figures or obsolete template assets')
    for p in (ROOT/'code/results').iterdir():
        if p.name!='README.md':add(p,'Retired pre-current-method experiment outputs, absent from every current evidence pin')
    keep_output={'README.md','pdf','optimized_results_20261026.json','optimized_results_20261026.md',
        'prototype_audit_two_reviews_20261027_sources.zip','prototype_audit_two_reviews_20261027_validation.json',
        'supervisor_methodology_20261027.zip','two_review_campaign_status_20261027.json','two_review_response_20261027.md',
        'two_review_results_20261027.json','two_review_tests_20261027.json'}
    for p in (ROOT/'output').iterdir():
        if p.name not in keep_output and not p.name.startswith('project_cleanup_'):add(p,'Superseded delivery, status report or duplicate artifact export')
    for p in (ROOT/'output/pdf').iterdir():
        if p.name not in ['prototype_audit_two_reviews_20261027.pdf','prototype_update_methodology_20261027.pdf']:
            add(p,'Superseded compiled paper; current full paper and methodology PDF retained')
    for p in (ROOT/'tmp').iterdir():
        if p.name=='pdfs':
            for q in p.iterdir():
                if q.name not in ['prototype_audit_two_reviews_20261027','supervisor_methodology_20261027']:
                    add(q,'Obsolete temporary build, render, extraction or packet-validation cache')
        else:add(p,'Obsolete temporary research/build workspace')
    for p in (ROOT/'code/.cache').iterdir():
        if p.name!='commitjl_data':add(p,'Retired download/recovery/report cache; current canonical datasets retained')
    keep_bins={'commitjl-bound-final-optimized-20261026.exe','full-plonk-control-referee-20261027.exe',
        'commitjl-bound-referee-final-20261025.exe','commitjl-bound-refinement-v2-20261025.exe'}
    for p in (ROOT/'code/zk/bin').iterdir():
        if p.is_file() and p.name not in keep_bins:add(p,'Obsolete test/experimental/Linux binary; current and selected replay binaries retained')
    keep_docs={'two_review_revision_20261027.md','group_certificate_protocol_20261024.md',
        'runtime_optimization_20261025.md','final_local_campaign_20261026.md',
        'referee_final_campaign_20261025.md','main_rank79_campaign_20261025.md'}
    for p in (ROOT/'code/docs').iterdir():
        if p.is_file() and p.name not in keep_docs:add(p,'Superseded narrative report or retired-server instructions')
    for p in (ROOT/'paper_results').iterdir():
        if p.is_file() and p.name!='README.md':add(p,'Unused early-paper table fragments and duplicate numerical exports')
    # Keys and compiled constraint/preparation caches are unnecessary for public
    # replay after a completed run. Preserve every byte in the live campaign and
    # the explicitly queued resumable online-affine control.
    for base in [ROOT/'code/validation/outputs',ROOT/'paper_results/review_controls']:
        for folder,dirs,files in os.walk(base):
            current=Path(folder)
            if current in candidates:dirs[:]=[];continue
            if any(current==p or current.is_relative_to(p) for p in [LIVE,AFFINE]):dirs[:]=[];continue
            for name in files:
                if Path(name).suffix.lower() in ['.pk','.r1cs','.gob']:
                    add(current/name,'Rebuildable proving/constraint/preparation cache from an inactive completed run; proofs, VKs, inputs and measured receipts retained')
    for base in [ROOT/'code',ROOT/'.pytest_cache']:
        if not base.exists():continue
        if base.name=='.pytest_cache':add(base,'Disposable pytest cache');continue
        for folder,dirs,files in os.walk(base):
            current=Path(folder)
            if any(current==p or current.is_relative_to(p) for p in [LIVE,AFFINE]):dirs[:]=[];continue
            for name in list(dirs):
                if name in ['__pycache__','.pytest_cache']:
                    add(current/name,'Disposable Python/test cache');dirs.remove(name)
                elif name in ['.git','.venv']:dirs.remove(name)
    # Collapse nested targets so each deleted file is counted once.
    selected=[]
    for p in sorted(candidates,key=lambda p:(len(p.parts),str(p))):
        if not any(p.is_relative_to(parent) for parent in selected):selected.append(p)
    records=[]
    for p in selected:
        count=0;size=0
        if p.is_file():count=1;size=p.stat().st_size
        else:
            for folder,dirs,files in os.walk(p):
                for name in dirs+files:
                    q=Path(folder)/name
                    if q.is_junction() or q.is_symlink():raise RuntimeError('Linked descendant in deletion tree: '+str(q))
                for name in files:
                    q=Path(folder)/name
                    try:size+=q.stat().st_size;count+=1
                    except FileNotFoundError:pass
        records.append(dict(path=p.relative_to(ROOT).as_posix(),absolute_path=str(p),reason=candidates[p],files=count,bytes=size,is_directory=p.is_dir()))
    live=[]
    for process in psutil.process_iter(['pid','name','cmdline']):
        try:
            args=process.info['cmdline'] or []
            if any('referee_two_reviews_20261027' in arg for arg in args):live.append(dict(pid=process.pid,name=process.info['name'],command=args))
        except psutil.Error:pass
    plan=dict(workspace=str(ROOT),created_unix=time.time(),authorization='User explicitly requested literal deletion of old, archive and fully useless project files.',
        deletion_not_archival=True,records=records,total_files=sum(v['files'] for v in records),total_bytes=sum(v['bytes'] for v in records),
        protected_prefixes=[str(p) for p in protected],protected_sha256={p.relative_to(ROOT).as_posix():value for p,value in pins.items()},
        active_processes_observed=live,disk_free_before=psutil.disk_usage(str(ROOT)).free,
        retained='Current paper/methodology, author input ZIPs, source code, canonical datasets, active/queued proof caches, public proofs, verification keys, native records and source receipts.')
    OUT.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print(json.dumps(dict(targets=len(records),files=plan['total_files'],gib=round(plan['total_bytes']/2**30,3),protected_hashes=len(pins),plan=str(OUT))))


if __name__=='__main__':main()
