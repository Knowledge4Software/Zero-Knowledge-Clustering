"""Prepare a curated source/evidence snapshot for owner review before publication."""
from pathlib import Path
import hashlib,json,re,shutil,zipfile

ROOT=Path(__file__).resolve().parents[2]


def main():
    out=ROOT/'output/review_public_artifact_20261022';out.mkdir(parents=True,exist_ok=False)
    files={}
    def add(source,name):
        if name in files:raise ValueError('duplicate artifact path '+name)
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);files[name]=source
    excluded={'.cache','results','outputs','__pycache__','bin','.venv','venv','node_modules','.git'}
    allowed={'.py','.go','.mod','.sum','.toml','.txt','.md','.yaml','.yml','.json','.csv'}
    for source in sorted((ROOT/'code').rglob('*')):
        rel=source.relative_to(ROOT/'code')
        if source.is_file() and not excluded.intersection(rel.parts) and (source.suffix in allowed or source.name in ['go.mod','go.sum']):
            add(source,'code/'+rel.as_posix())
    # Preserve original numerical reproduction evidence. It comes from public
    # benchmarks; diagnostic labels/flags are separate from protocol exports.
    for folder in ['review_final_capture_20261020_1907Z','review_full_capture_followup_20261020',
                   'review_owner_inputs_20261020','review_certificate_preflight_20261020','review_certificate_budget_20261020']:
        base=ROOT/'code/validation/outputs'/folder
        for source in sorted(base.rglob('*')):
            if source.is_file() and source.suffix in {'.json','.csv'}:add(source,source.relative_to(ROOT).as_posix())
    for source in sorted((ROOT/'paper_results').rglob('*')):
        if source.is_file() and source.suffix in {'.json','.csv'}:add(source,source.relative_to(ROOT).as_posix())
    new=ROOT/'code/validation/outputs/review_update_decisions_20261021T184553Z/native_capture_20261022T0530Z'
    for source in sorted(new.rglob('*')):
        if source.is_file():add(source,'new_evidence/'+source.relative_to(new).as_posix())
    for target,name in [('commitjl-bound-paired-20261021.exe','commitjl-bound.exe'),('commitjl-bound-paired-20261021-linux','commitjl-bound-linux')]:
        add(ROOT/'code/zk/bin'/target,'bin/'+name)
    cases=[]
    def public_case(name,source,pin):
        prefix='public_cases/'+name
        for p in sorted((source/'public_bundle').rglob('*')):
            if p.is_file():
                if p.suffix not in {'.json','.vk','.proof'}:raise ValueError('non-public file type in export')
                if p.suffix=='.json':
                    d=json.loads(p.read_text())
                    if any('assign' in s or 'certified_flags' in s for s in d.get('samples') or []):raise ValueError('private diagnostics in public export')
                add(p,prefix+'/bundle/'+p.relative_to(source/'public_bundle').as_posix())
        add(source/pin,prefix+'/owner_pin.json');add(source/'approved_public_task.json',prefix+'/approved_task.json')
        cases.append(dict(name=name,bundle=prefix+'/bundle',pin=prefix+'/owner_pin.json',task=prefix+'/approved_task.json'))
    for cohort in ['original_full','original_reuse']:
        for summary in sorted((ROOT/'tmp/review_public_artifact_20261022'/cohort).glob('*/summary.json')):
            public_case(cohort+'_'+summary.parent.name,summary.parent,'owner_setup_pin.json')
    if len(cases)!=14:raise ValueError('all fourteen original proof exports required')
    release=ROOT/'code/validation/outputs/review_paired_release_20261021'
    public_case('paired_synthetic',release/'decision_smoke','owner_pin_paired_change.json')
    public_case('full_synthetic',release/'decision_full_smoke','owner_pin_full_reuse.json')
    current=ROOT/'code/validation/outputs/review_update_decisions_20261021T184553Z/public_capture_20261022T0540Z'
    for name in ['mnist_k3_seed11','fashion_mnist_k5_seed11']:
        public_case(name+'_paired',current/(name+'_learned_r63'),'owner_pin_paired_change.json')
        public_case(name+'_full',current/(name+'_full_r0'),'owner_pin_full_reuse.json')
    # Check only credential-shaped secrets, not public synthetic commitment salts.
    pattern=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
    for name in files:
        p=out/name
        if p.suffix not in {'.exe','.vk','.proof'} and pattern.search(p.read_bytes()):raise ValueError('credential-shaped material: '+name)
    readme='''# Review artifact - staged, not yet published

This package contains code, public-benchmark numerical evidence and public-only
proof exports for verifiable prototype-update audits. It is an author-review
snapshot. Public repository publication requires owner approval. It does not
claim that the private development repository is publicly accessible.

## Verify without datasets or private witnesses

Use Python3 and the bundled verifier for your platform:

```
python code/validation/verify_public_artifact.py --root .
```

The script checks every packaged hash, copies each public bundle to a fresh
verification_runs directory, and performs new cryptographic verification using
separately packaged owner pins and approved tasks. Original evidence is not
modified. No original row, private assignment or witness is needed for this
verification. Owner pins model the experiment's trusted owner/setup; a real
deployment must obtain authentic pins/tasks independently of the prover.

To build the verifier from the included source instead:

```
go -C code/zk build -trimpath -buildvcs=false -o ../../bin/local-verifier ./cmd/commitjl-bound
python code/validation/verify_public_artifact.py --root . --bin bin/local-verifier
```

On Windows give the executable an .exe suffix. Dependencies are pinned by the
Go module files. Python experiment dependencies are in code/requirements.txt.
Run Python tests with PYTHONPATH=code and pytest code/tests, excluding the optional
legacy test_zk.py integration module unless its external binaries are installed.

## Evidence and limitations

The fourteen original completed workloads and their public proofs are retained.
New evidence includes the completed2160-cell native independent-fit grid and
selected completed128-record proof examples. The full new proof campaign is
still running; these examples do not establish a universal success rate.
Real-data traces are Lloyd optimization snapshots, not production update logs.

Numerical reproduction inputs and diagnostic records under code/validation/outputs
come from public benchmark datasets. They may contain benchmark labels/flags for
offline checks. They are distinct from public_cases/, whose protocol exports
contain no private witnesses or individual assignments.

The review response and exact paired-bound derivation are documented in
code/docs/review_response_20261021.md. The native grid records failures and
inconclusive outcomes, including the text workloads. Known geometric identities
and proof primitives are attributed; no exhaustive novelty or acceptance claim
is made. No license change is imposed by this staged package; authors decide
publication and distribution terms before release.
'''
    (out/'README.md').write_text(readme,encoding='utf-8')
    records=[dict(path=p.relative_to(out).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(out.rglob('*')) if p.is_file()]
    manifest=dict(version='review-artifact-20261022',publication='pending owner approval',files=records,public_cases=cases,
        go_source_files_byte_preserved=True,private_protocol_witnesses_in_public_cases=False,credential_pattern_check='PASS')
    (out/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    archive=ROOT/'output/review_public_artifact_20261022.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out).as_posix())
    receipt=dict(directory=str(out),archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                 files=len(records),public_cases=len(cases),publication='not published')
    (ROOT/'output/review_public_artifact_20261022_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
