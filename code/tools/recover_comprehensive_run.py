"""Verify frozen results and quarantine only invalid checkpoints after power loss."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import psutil

from commitjl.durable_io import atomic_json
from commitjl.reusable_native import NativeProjection,file_hash

CODE=Path(__file__).resolve().parents[1]


def invalidate_dependent_fits(root,bad_fits):
    """A relearned center set requires relearning both bases for that start."""
    expanded=dict(bad_fits)
    for name in bad_fits:
        path=Path(name)
        if path.stem.startswith('centers_s'):
            suffix=path.stem[len('centers_'):]
            for kind in ['basis','enriched']:
                dependent=str(path.with_name(f'{kind}_{suffix}.json'))
                if (root/dependent).exists():
                    expanded.setdefault(dependent,'Invalidated because upstream center fit is damaged: '+name)
    return expanded


def active_workers(root):
    found=[]
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            cmd=' '.join(process.info['cmdline'] or [])
            if str(root).lower() in cmd.lower() and any(name in cmd for name in [
                'campaigns.run_comprehensive_native','experiments.comprehensive_native_comparison',
                'campaigns.comprehensive_durable_worker']):found.append(process.info['pid'])
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return found


def recovered_identity(root,reference):
    from experiments import comprehensive_native_comparison as module
    saved=json.loads((reference/'manifest.json').read_text())
    sources=saved['identity']['sources']
    for relative,digest in sources.items():
        if file_hash(root/'sources'/relative)!=digest or file_hash(CODE/relative)!=digest:
            raise ValueError('Frozen and current source differ: '+relative)
    archive=CODE/'results/server_retirement_20261007/compact_arrays/array_index.json'
    if file_hash(archive)!=saved['identity']['archive_index_sha256']:raise ValueError('Archive index changed')
    captured={};original_run=module.run;original_argv=sys.argv
    try:
        module.run=lambda args:captured.update(vars(args))
        sys.argv=['recover','--out',str(root),'--max-seconds','28800','--max-rss-gib','2.5']
        module.main()
    finally:module.run=original_run;sys.argv=original_argv
    captured['out']=str(root)
    captured.update(q=1,extra=12,version='comprehensive-native-v1')
    return dict(configuration=captured,sources=sources,
        archive_index_sha256=saved['identity']['archive_index_sha256'])


def recover(root,reference,repair=False):
    root=root.resolve();reference=reference.resolve()
    if active_workers(root):raise RuntimeError('A campaign is active; recovery will not touch it')
    identity=recovered_identity(root,reference)
    records={};bad={};references=defaultdict(set);expected_hashes={}
    for path in sorted((root/'rows').glob('*.json')):
        relative=str(path.relative_to(root))
        try:row=json.loads(path.read_text())
        except (ValueError,UnicodeError) as error:
            bad[relative]=f'Invalid case JSON: {error}';continue
        records[relative]=row
        for name,digest in row['artifacts'].items():
            if name in expected_hashes and expected_hashes[name]!=digest:
                raise ValueError('Conflicting recorded artifact hashes: '+name)
            expected_hashes[name]=digest;references[name].add(relative)
    bad_artifacts={}
    for i,(name,digest) in enumerate(expected_hashes.items(),1):
        path=root/name
        actual=file_hash(path) if path.exists() else None
        if actual!=digest:
            bad_artifacts[name]=dict(expected=digest,actual=actual)
            for case in references[name]:bad[case]='Referenced artifact checksum mismatch: '+name
        if i%2000==0:print(json.dumps(dict(event='artifact_checks',done=i,total=len(expected_hashes))),flush=True)
    bad_fits={};fit_count=0
    for path in (root/'datasets').rglob('*.json'):
        if path.parent.name!='models':continue
        fit_count+=1
        try:
            meta=json.loads(path.read_text());binary=path.with_suffix('.npz')
            if file_hash(binary)!=meta['sha256']:raise ValueError('Fit array checksum mismatch')
        except (ValueError,UnicodeError,OSError,KeyError) as error:
            bad_fits[str(path.relative_to(root))]=str(error)
    bad_fits=invalidate_dependent_fits(root,bad_fits)
    for name in bad_fits:
        p=Path(name);dataset=p.parts[1];stem=p.stem
        for case,row in records.items():
            if row['dataset']!=dataset:continue
            if p.parts[2].startswith('k'):
                if row['k']!=int(p.parts[2][1:]) or f"_s{row['start']}" not in stem:continue
                depends=(stem.startswith('centers_') or
                    stem.startswith('enriched_') and row['method']=='protected_plus12' or
                    stem.startswith('basis_') and row['method'] in ['protected_only','protected_jl'])
            else:depends=stem==f"{row['method']}_m{row['m']}_seed{row['seed']}"
            if depends:bad[case]='Invalid dependent fit: '+name
    # A partially written case may have created a model before committing its
    # row. Validate those unreferenced models too, since the runner reuses them.
    bad_models={}
    for path in (root/'projection_models').glob('*.npz'):
        name=str(path.relative_to(root))
        if name in expected_hashes:continue
        try:
            with np.load(path,allow_pickle=False) as arrays:
                a={key:arrays[key] for key in arrays.files}
            d=len(a['origin'])
            if 'basis' in a:
                from scipy.sparse import csr_matrix
                sparse=csr_matrix((a['a_data'],a['a_indices'],a['a_indptr']),shape=tuple(a['a_shape']))
                model=NativeProjection('hybrid',d,basis=a['basis'],sparse_map=sparse,origin=a['origin'])
            elif 'components' in a:model=NativeProjection('pca',d,components=a['components'],origin=a['origin'])
            else:model=NativeProjection('full',d,origin=a['origin'])
            if model.fingerprint!=path.stem:raise ValueError('Model fingerprint mismatch')
        except Exception as error:bad_models[name]=f'{type(error).__name__}: {error}'
    good=[case for case in records if case not in bad]
    planned=json.loads((root/'plan.json').read_text())
    result=dict(complete=True,repaired=False,verified_cases=len(good),parsed_cases=len(records),
        expected_cases=planned['expected_cases'],unique_artifacts_checked=len(expected_hashes),fit_caches_checked=fit_count,
        invalid_cases=bad,invalid_artifacts=bad_artifacts,invalid_fits=bad_fits,invalid_unreferenced_models=bad_models,
        source_hashes_verified=len(identity['sources']),identity=identity,
        scope='All committed-case artifact checksums and fit-cache checksums, plus unreferenced model replay')
    if repair:
        if active_workers(root):raise RuntimeError('A campaign started during recovery')
        folder=root/f'recovery_{time.time_ns()}';folder.mkdir()
        def quarantine(relative):
            source=root/relative
            if not source.exists():return
            target=folder/'quarantine'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(source),str(target))
        for relative in bad:quarantine(relative)
        for relative in bad_artifacts:quarantine(relative)
        for relative in bad_models:quarantine(relative)
        for relative in bad_fits:
            quarantine(relative);quarantine(str(Path(relative).with_suffix('.npz')))
        # Orphan fit arrays lacking a valid metadata record are overwritten by
        # cached_fit; no valid fit or complete case is removed unnecessarily.
        if (root/'manifest.json').exists():shutil.copyfile(root/'manifest.json',folder/'damaged_manifest.json')
        for name in ['launcher.json','background_status.json']:
            if (root/name).exists():shutil.copyfile(root/name,folder/name)
        elapsed=0.
        for line in (root/'run.log').read_text(encoding='utf-8').splitlines():
            try:elapsed=max(elapsed,float(json.loads(line).get('elapsed',0)))
            except (ValueError,TypeError):pass
        manifest=dict(identity=identity,complete=False,completed_cases=len(good),expected_cases=planned['expected_cases'],
            elapsed_seconds=elapsed,recovered_after_shutdown=True,recovery_report=str(folder/'recovery.json'))
        atomic_json(root/'manifest.json',manifest)
        result.update(repaired=True,recovery_directory=str(folder),interrupted_elapsed_seconds=elapsed)
        atomic_json(folder/'recovery.json',result);atomic_json(root/'recovery_latest.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['identity','invalid_cases']},indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--repair',action='store_true')
    args=parser.parse_args();recover(args.root,args.reference,args.repair)


if __name__=='__main__':main()
