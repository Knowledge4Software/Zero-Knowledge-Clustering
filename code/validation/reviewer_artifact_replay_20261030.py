"""Standalone small proof replay and optional fresh synthetic proof generation.

Requires Python 3.12 standard library and the binary built from code/zk.
No training datasets, NumPy, private proof archive or server is required.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / 'reviewer_artifact/fixtures'
FIELD = 21888242871839275222246405745257275088548364400416034343698204186575808495617


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_cost(rows, centers):
    return sum(min(sum((int(a)-int(b))**2 for a, b in zip(row, c))
                   for c in centers) for row in rows)


def validate_decisions(decisions, instance):
    assert [r['set'] for r in decisions] == list(range(len(instance['reuse_centers'])))
    truth = []
    for r, centers in zip(decisions, instance['reuse_centers']):
        change = exact_cost(instance['x'], centers)-exact_cost(instance['x'], instance['comparison_base'])
        value = change*int(r['denominator'])
        assert int(r['lower']) <= value <= int(r['upper'])
        expected = 'improvement' if change < 0 else 'worsening' if change > 0 else 'equal'
        assert r['decision'] in [expected, 'inconclusive']
        truth.append(change)
    return truth


def invoke(binary, phase, target, trust, task, work, *, base=None,
           refine_trust=None, instance=None, expected_success=True):
    args = [str(binary), '-phase', phase, '-out', str(target), '-trust', str(trust),
            '-public-task', str(task), '-q', '2', '-reps', '1', '-warmup', '0',
            '-max-constraints', '1600000']
    if base is not None:
        args += ['-refinement-base', str(base), '-refinement-trust', str(refine_trust)]
    if instance is not None:
        args += ['-instance', str(instance)]
    env = dict(os.environ, GOMAXPROCS='2', GOMEMLIMIT='512MiB', COMMITJL_RESUME_PROOFS='1')
    result = subprocess.run(args, cwd=work, env=env, capture_output=True, text=True,
                            encoding='utf-8', errors='replace', timeout=240)
    assert (result.returncode == 0) == expected_success, result.stdout+result.stderr
    return result.returncode


def export_public(private, public):
    public.mkdir()
    names = ['group_commit_receipt.json', 'group_public_rows.json', 'group_rows.json',
             'group_query.json', 'group_query_records.json']
    names += [p.relative_to(private).as_posix() for p in (private/'keys').glob('*.vk')]
    for phase in ['group_rows', 'group_query']:
        names += [p.relative_to(private).as_posix() for p in (private/phase).glob('*.proof')]
    for name in names:
        dest = public/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(private/name, dest)
    save(public/'public_manifest.json', dict(sha256={n: sha(public/n) for n in names}))


def replay(binary, work):
    """Only public bundles and pins are copied into the verifier work directory."""
    cases = []
    for name in ['compact', 'refinement']:
        source = FIXTURES/name
        target = work/name
        target.mkdir()
        # Raw input is consulted by the independent oracle below, never passed
        # to either Go verification phase or copied into this verifier directory.
        instance, expected = read(source/'input.json'), read(source/'expected.json')
        for filename in ['approved.json', 'owner_pin.json']:
            shutil.copy2(source/filename, target/filename)
        shutil.copytree(source/'public', target/'public')
        base = None
        if name == 'refinement':
            shutil.copytree(source/'base_public', target/'base_public')
            shutil.copy2(source/'refinement_owner_pin.json', target/'refinement_owner_pin.json')
            base = target/'base_public'
            invoke(binary, 'group_verify', base, target/'owner_pin.json', target/'approved.json', target)
            assert read(base/'group_verified_decisions.json') == expected['base_decisions']
        phase = 'group_verify' if base is None else 'refinement_verify'

        def verify(ok=True):
            return invoke(binary, phase, target/'public', target/'owner_pin.json',
                          target/'approved.json', target, base=base,
                          refine_trust=target/'refinement_owner_pin.json', expected_success=ok)

        verify()
        record = 'group_verified_decisions.json' if base is None else 'refinement_verified_decisions.json'
        decisions = read(target/'public'/record)
        assert decisions == expected['decisions']
        truth = validate_decisions(decisions, instance)
        assert truth == expected['exact_changes']
        receipt = read(target/'public'/(phase+'.json'))
        assert receipt['verified'] == expected['verified_proofs']
        rejected = []

        def mutate_json(path, mutate, label):
            original = path.read_bytes()
            changed = read(path)
            mutate(changed)
            save(path, changed)
            try:
                verify(False)
                rejected.append(label)
            finally:
                path.write_bytes(original)

        proof = sorted((target/'public').rglob('*.proof'))[0]
        before = proof.read_bytes()
        proof.write_bytes(bytes([before[0] ^ 1])+before[1:])
        try:
            verify(False)
            rejected.append('damaged proof')
        finally:
            proof.write_bytes(before)
        mutate_json(target/'approved.json',
                    lambda v: v['reuse_centers'][1][0].__setitem__(0, 3), 'changed approved center')
        if base is None:
            mutate_json(target/'public/group_public_rows.json', lambda v: v.pop(), 'missing row coverage')
        else:
            def duplicate(v):
                selected = next(r for r in v if r['samples'])
                selected['samples'].append(copy.deepcopy(selected['samples'][0]))
            mutate_json(target/'public/refinement_records.json', duplicate, 'duplicate refinement row')
        verify()
        assert not list(target.rglob('*.pk')) and not list(target.rglob('*.gob'))
        assert not list(target.rglob('input.json'))
        cases.append(dict(case=name, verified_proofs=receipt['verified'],
                          exact_changes=truth, decisions=decisions, rejected_mutations=rejected,
                          no_instance_or_private_cache_in_verifier_directory=True))
    return cases


def fresh(binary, work):
    """Create a complete tiny compact path with new keys, proofs and challenge."""
    work.mkdir()
    private, public = work/'prover', work/'public'
    input_path = work/'input.json'
    instance = read(FIXTURES/'compact/input.json')
    save(input_path, instance)
    task = {k: instance[k] for k in ['queries', 'reuse_centers', 'comparison_base']}
    save(work/'approved.json', task)
    pin = work/'owner_pin.json'
    invoke(binary, 'group_compact_prepare', private, pin, work/'approved.json', work,
           instance=input_path)
    receipt, trust = read(private/'group_commit_receipt.json'), read(private/'trusted.json')
    assert receipt['version'] == 'group-compact-v3'
    assert trust['original_root'] == receipt['statement']['cx']
    owner = dict(version=receipt['version'], owner=trust, statement=receipt['statement'],
                 header=dict(GroupRoot=receipt['GroupRoot'], StatsRoot=receipt['StatsRoot'],
                             Challenge=secrets.randbelow(FIELD-1)+1),
                 commit_sha256=sha(private/'group_commit_receipt.json'), q=2)
    save(pin, owner)
    invoke(binary, 'group_keys', private, pin, work/'approved.json', work)
    owner['owner'] = read(private/'trusted.json')
    save(pin, owner)
    for phase in ['group_rows', 'group_query']:
        invoke(binary, phase, private, pin, work/'approved.json', work)
    export_public(private, public)
    invoke(binary, 'group_verify', public, pin, work/'approved.json', work)
    decisions = read(public/'group_verified_decisions.json')
    expected = read(FIXTURES/'compact/expected.json')
    assert decisions == expected['decisions']
    assert validate_decisions(decisions, instance) == expected['exact_changes']
    assert read(public/'group_verify.json')['verified'] == 13
    return dict(complete=True, fresh_setup=True, fresh_post_commit_challenge=True,
                fresh_proofs=13, decisions=decisions,
                role_scope='local simulation, not authenticated production challenge delivery',
                performance_benchmark=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()
    binary = args.binary.resolve()
    assert binary.is_file(), binary
    if args.out:
        work = args.out.resolve()
        work.mkdir(parents=True, exist_ok=False)
    else:
        work = Path(tempfile.mkdtemp(prefix='prototype-reviewer-'))
    cases = replay(binary, work)
    new = fresh(binary, work/'fresh') if args.fresh else None
    result = dict(complete=True, cases=cases, fresh=new, binary_sha256=sha(binary),
                  new_performance_measurements=False, output_directory=str(work))
    save(work/'reviewer_checks.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
