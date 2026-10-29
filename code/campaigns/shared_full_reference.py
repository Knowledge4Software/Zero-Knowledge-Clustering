"""Recoverably share an already completed exact Full baseline reference.

Copying these files is never fresh baseline or fallback proof generation.
Existing matching files remain untouched; conflicting files/directories are
retained beside their original path before the authoritative source is copied.
"""
from pathlib import Path
import json,os,shutil,time
from commitjl.pilot_io import save,sha
from campaigns.proof_runtime import check_reference_runtime


def read(path):return Path(path).read_text(encoding='utf-8')


def _inside(path,owner):
    resolved=path.resolve()
    if not resolved.is_relative_to(owner) or resolved==owner:
        raise ValueError(f'reference path escapes its owned directory: {path}')
    return resolved


def _validate_source(source,inp):
    state=json.loads(read(source/'status.json'))
    if not state.get('cases',{}).get('full',{}).get('complete'):
        raise ValueError('source Full baseline is not complete')
    if state.get('input_sha256',{}).get('full')!=sha(source/'full/input.json'):
        raise ValueError('source Full input does not match its recorded pin')
    n=inp['model']['n'];queries=inp['queries'];expected=len(queries)*len(inp['reuse_centers'])
    if inp['model']['method']!='full' or sorted(i for q in queries for i in q)!=list(range(n)):
        raise ValueError('source must be a complete exact Full catalog')
    full=source/'full';phase=f'full_reuse_{len(queries[0])}.json'
    generated=json.loads(read(full/'bundle'/phase));replay=json.loads(read(full/'public_bundle/verify.json'))
    if not generated.get('complete') or generated.get('verified')!=expected or len(generated.get('samples',[]))!=expected:
        raise ValueError('source Full generation receipt is incomplete')
    if not replay.get('complete') or replay.get('verified')!=expected:
        raise ValueError('source Full public replay is incomplete')
    for name in ['prepare','full_reuse','verify']:
        if not json.loads(read(full/(name+'.resource.json'))).get('complete'):
            raise ValueError(f'source Full resource receipt is incomplete: {name}')
    public=(full/'public_bundle').resolve();manifest=json.loads(read(public/'public_manifest.json'))
    hashes=manifest.get('sha256',{})
    if not hashes:raise ValueError('source Full public manifest is empty')
    for name,digest in hashes.items():
        path=_inside(public/name,public)
        if not path.is_file() or sha(path)!=digest:raise ValueError(f'source Full public file hash mismatch: {name}')
    return state


def share_full_reference(source,target):
    source=Path(source).resolve();target=Path(target).resolve()
    if source==target:raise ValueError('shared reference source and target must differ')
    # A reference copy cannot turn a two-worker baseline into an eight-worker
    # measurement. Historical fixtures/receipts without a field use two.
    protocols=[json.loads(read(root/'protocol.json')) if (root/'protocol.json').exists() else {} for root in [source,target]]
    check_reference_runtime(*protocols)
    inp=json.loads(read(source/'full/input.json'))
    if json.loads(read(target/'full/input.json'))!=inp:
        raise ValueError('shared Full inputs differ; no reference files were copied')
    state=_validate_source(source,inp)
    copied=[];archived=[];matching=[]
    def archive(path):
        _inside(path,target)
        candidate=path.with_name(path.name+f'.partial_{os.getpid()}_{time.time_ns()}')
        _inside(candidate,target)
        if candidate.exists():raise FileExistsError(candidate)
        path.rename(candidate);archived.append(candidate.relative_to(target).as_posix())
    def directory(path):
        if path==target:return
        _inside(path,target)
        directory(path.parent)
        if path.exists() and not path.is_dir():archive(path)
        path.mkdir(exist_ok=True)
    def copy_file(path,destination):
        _inside(destination,target);directory(destination.parent)
        digest=sha(path)
        if destination.is_file() and sha(destination)==digest:
            matching.append(destination.relative_to(target).as_posix());return
        if destination.exists():archive(destination)
        shutil.copy2(path,destination)
        if sha(destination)!=digest:raise RuntimeError(f'incomplete shared reference copy: {destination}')
        copied.append(destination.relative_to(target).as_posix())
    full=(source/'full').resolve()
    # Parent-first ordering repairs interrupted directory copies as well as
    # missing/corrupt individual files. Unrelated target evidence is retained.
    for path in sorted(full.rglob('*'),key=lambda p:(len(p.parts),p.as_posix())):
        if path.is_symlink():raise ValueError(f'source reference contains a symlink: {path}')
        relative=path.relative_to(full)
        if relative==Path('input.json'):continue
        destination=target/'full'/relative
        if path.is_dir():directory(destination)
        elif path.is_file():copy_file(path,destination)
    copy_file(source/'status.json',target/'status.json')
    provenance_path=target/'full_reference_provenance.json'
    provenance={}
    if provenance_path.exists():
        try:provenance=json.loads(read(provenance_path))
        except (ValueError,OSError):archive(provenance_path)
    receipt=dict(source_status_sha256=sha(source/'status.json'),
        source_public_manifest_sha256=sha(source/'full/public_bundle/public_manifest.json'),
        copied_files=copied,matching_files=len(matching),archived_paths=archived,
        checked_unix=time.time(),new_proof_generation=False)
    provenance.update(version='shared-full-reference-v1',source=str(source/'full'),
        scope='Same completed exact Full baseline reference shared by paired methods; not newly generated PCA or fallback proofs.',
        input_sha256=sha(source/'full/input.json'),target_input_sha256=sha(target/'full/input.json'))
    provenance.setdefault('copy_attempts',[]).append(receipt);save(provenance_path,provenance)
    return receipt
