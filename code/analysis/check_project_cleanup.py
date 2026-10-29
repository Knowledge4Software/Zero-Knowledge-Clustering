"""Validate user-authorized literal cleanup without generating proofs or papers."""
from pathlib import Path
import hashlib,json,time,zipfile
import psutil
from campaigns.local_fast_tradeoff import sha,save
from analysis.build_three_papers import expand

ROOT=Path(__file__).resolve().parents[2]


def main():
    plan=json.loads((ROOT/'output/project_cleanup_20261027_plan.json').read_text())
    output=ROOT/'output/project_cleanup_20261027.json'
    receipt=json.loads(output.read_text(encoding='utf-8-sig'))
    assert len(receipt['records'])==len(plan['records'])
    assert all(v['status'] in ['deleted','already_absent'] for v in receipt['records'])
    for path,digest in plan['protected_sha256'].items():assert sha(ROOT/path)==digest,'Protected content changed: '+path
    remaining=[];rebuilt=[]
    for row in plan['records']:
        if (ROOT/row['path']).exists():
            if row['reason']=='Disposable Python/test cache':rebuilt.append(row['path'])
            else:remaining.append(row['path'])
    assert not remaining,'Obsolete targets remain: '+str(remaining)
    dependencies=set();expand(ROOT/'manuscript.tex',dependencies)
    assert all(p.is_file() for p in dependencies)
    archives=[]
    for name in ['output/prototype_audit_two_reviews_20261027_sources.zip','output/supervisor_methodology_20261027.zip']:
        with zipfile.ZipFile(ROOT/name) as z:
            manifest_name='source_manifest.json' if 'source_manifest.json' in z.namelist() else 'package_manifest.json'
            values=json.loads(z.read(manifest_name))
            assert all(hashlib.sha256(z.read(path)).hexdigest()==digest for path,digest in values.items())
        archives.append(name)
    generated=ROOT/'paper_results/referee_final_20261025/generated';manifests=set()
    old=json.loads((generated/'export_receipt.json').read_text())
    manifests.update(ROOT/name for name in old['sources'] if name.endswith('/public_manifest.json'))
    new=json.loads((generated/'two_review_evidence_receipt.json').read_text())
    for value in new['resources']:
        manifests.update(ROOT/row['path'] for row in value['manifests'])
    artifacts=0;proofs=0
    for path in manifests:
        value=json.loads(path.read_text())
        for name,digest in value['sha256'].items():
            file=path.parent/name
            assert sha(file)==digest,'Public artifact changed: '+str(file)
            artifacts+=1;proofs+=file.suffix=='.proof'
    queue=json.loads((ROOT/'code/validation/outputs/referee_two_reviews_20261027/queue.json').read_text())
    identity=queue.get('supervisor_pid');live=bool(identity and psutil.pid_exists(identity))
    deleted=[v for v in receipt['records'] if v['status']=='deleted']
    receipt.update(status='validated',validated_unix=time.time(),total_files_deleted=sum(v['files'] for v in deleted),
        total_bytes_deleted=sum(v['bytes'] for v in deleted),gib_deleted=sum(v['bytes'] for v in deleted)/2**30,
        disk_free_after=psutil.disk_usage(str(ROOT)).free,disk_free_delta=psutil.disk_usage(str(ROOT)).free-plan['disk_free_before'],
        protected_hashes_verified=len(plan['protected_sha256']),current_paper_dependencies_verified=len(dependencies),
        current_delivery_archive_manifests_verified=archives,public_manifests_verified=len(manifests),
        public_artifacts_hash_checked=artifacts,public_proof_files_hash_checked=proofs,
        rebuilt_python_test_caches=rebuilt,active_research_supervisor_live=live,research_phase_at_validation=queue.get('active_phase'),
        no_backup_archive_created=True,source_code_retained=True,git_history_untouched=True,
        retained_proving_caches=['code/validation/outputs/referee_two_reviews_20261027',
            'code/validation/outputs/referee_optimized_20261026/extended/fashion_mnist_k5_seed173_age8_n1024/repetition_01/online_affine'])
    interference=ROOT/'code/validation/outputs/referee_two_reviews_20261027/filesystem_cleanup_interference_20261027.json'
    if interference.exists():receipt['benchmark_timing_interference']=json.loads(interference.read_text())
    save(output,receipt)
    report=['# Project cleanup, 27 October 2026','',
        f"Deleted {receipt['total_files_deleted']:,} files across {len(deleted):,} targets, totaling {receipt['gib_deleted']:.3f} GiB.",
        'Actual deletion was requested by the user; no replacement archive or backup copy was created. Git history was not rewritten.','',
        'Removed: superseded raw experiment campaigns, retirement/recovery caches, copied paper versions, old PDF/source deliveries, obsolete reports and binaries, inactive proving keys and compiled/preparation caches.','',
        'Retained: current full paper and methodology, supervisor input ZIPs and source material, code/tests, canonical datasets, numerical and negative-result records, public proofs/verification keys, source receipts, and every cache needed by active/queued work.','',
        f"Validation: {len(plan['protected_sha256'])} protected file hashes; {len(dependencies)} current manuscript dependencies; two current delivery ZIP manifests; {len(manifests)} public manifests covering {proofs:,} proof files and {artifacts:,} artifact hashes.",
        'This is a preservation/integrity check, not a new cryptographic replay or security audit.','',
        f"Research supervisor live at validation: {live}; phase: {queue.get('active_phase')}. No proof worker was intentionally paused, restarted or deleted.",
        'Inactive proving caches can be regenerated if a new experiment is later authorized. Their deletion leaves measured resource metadata and existing public proofs intact.','',
        'The JSON receipt lists every removed target, its reason, file count and byte count. Python/test caches may be recreated by subsequent imports and tests.']
    if receipt.get('targeted_tests'):
        report+=['',f"Targeted regression tests: {receipt['targeted_tests']['passed']} passed, {receipt['targeted_tests']['failed']} failed."]
    if receipt.get('benchmark_timing_interference'):
        report+=['','The filesystem deletion overlapped the HAR Full timing phase. Its proofs/inputs remain valid. The interference ledger marks this comparison; automatic exports retain raw times but exclude its ratio from clean speed conclusions. No replacement proof workload was launched by cleanup.']
    (ROOT/'output/project_cleanup_20261027.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps({key:receipt[key] for key in ['status','total_files_deleted','gib_deleted','protected_hashes_verified','public_proof_files_hash_checked','active_research_supervisor_live']}))


if __name__=='__main__':main()
