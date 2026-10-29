"""Validate the new private release and bind a completed visual inspection."""
from pathlib import Path
import argparse,json,subprocess,zipfile
from campaigns.local_fast_tradeoff import save,sha
from analysis.check_round3_release import numeric_cells

REPO=Path(__file__).resolve().parents[2]
TAG='prototype_audit_optimized_20261026'


def main(reviewed_render,tag=TAG):
    path=REPO/'output'/(tag+'_validation.json');v=json.loads(path.read_text())
    assert sha(REPO/v['pdf'])==v['pdf_sha256']
    source=REPO/v['source_zip'];assert sha(source)==v['source_zip_sha256']
    with zipfile.ZipFile(source) as archive:
        import hashlib
        manifest=json.loads(archive.read('source_manifest.json'))
        assert all(hashlib.sha256(archive.read(name)).hexdigest()==digest for name,digest in manifest.items())
        assert 'code/validation/outputs/referee_optimized_20261026/native/records.json' in manifest
    names=subprocess.check_output(['git','ls-tree','-r','--name-only','85da2c1','paper_results/referee_final_20261025/generated'],cwd=REPO,text=True).splitlines()
    tables=[name for name in names if name.endswith('.tex')]
    for name in tables:
        previous=subprocess.check_output(['git','show','85da2c1:'+name],cwd=REPO,text=True)
        assert numeric_cells(previous)==numeric_cells((REPO/name).read_text()),'Historical numbers changed: '+name
    reviewed=reviewed_render;current=REPO/v['render_directory']
    assert len(list(current.glob('page-*.png')))==v['pages']
    assert len(list(reviewed.glob('page-*.png')))==v['pages']
    assert all(sha(image)==sha(reviewed/image.name) for image in current.glob('page-*.png')),'Reinspect changed pages'
    receipt=json.loads((REPO/'paper_results/referee_final_20261025/generated/optimized_evidence_receipt.json').read_text())
    assert all(sha(REPO/name)==digest for name,digest in receipt['source_sha256'].items())
    timing=json.loads((REPO/'paper_results/referee_final_20261025/generated/optimized_timing_receipt.json').read_text())
    v.update(status='ready_for_private_supervisor_review',visual_review='PASS',visual_review_pages=list(range(1,v['pages']+1)),
        visual_review_method='All rendered pages inspected in contact sheets; new result table pages additionally inspected at full resolution. Final render hashes match the explicitly supplied inspected render.',
        historical_numeric_tables_unchanged=len(tables),source_archive_manifest_verified=True,
        native_representation_fits=56,native_records=280,large_scale_proof_timings_pending=True,
        new_matched_repetitions_complete=timing['completed_repetitions'],completed_new_method_full_pairs=timing['completed_method_pairs'],all_requested_extended_proofs_complete=False,
        unresolved_scientific_issues=True,submission_ready=False)
    scope=receipt.get('user_execution_scope')
    if scope:v.update(maximum_proof_catalog_records=scope['maximum_proof_catalog_records'],
        large_scale_proof_timings_pending=False,large_scale_proof_timings_deferred=True)
    if v.get('new_evidence_receipt_sha256'):
        new_path=REPO/'paper_results/referee_final_20261025/generated/two_review_evidence_receipt.json'
        assert sha(new_path)==v['new_evidence_receipt_sha256'],'New evidence changed after PDF build'
        new=json.loads(new_path.read_text())
        assert all(sha(REPO/name)==digest for name,digest in new['source_sha256'].items())
        v.update(new_review_evidence_pins_verified=True,
            completed_new_backend_executions=len(new['completed_plonk_runs']),
            completed_new_learning_proof_pairs=len(new.get('new_proof_pairs',[])))
    save(path,v);print(json.dumps(v,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reviewed-render',type=Path,default=REPO/'tmp/pdfs'/TAG/'renders/6e80fe544f73');p.add_argument('--tag',default=TAG);a=p.parse_args();main(a.reviewed_render.resolve(),a.tag)
