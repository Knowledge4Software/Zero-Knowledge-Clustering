"""Public-only verifier for disjoint compact/refinement and exact Full paths.

The caller supplies an independently approved task and owner/key pins. Routing
is untrusted. This wrapper freshly invokes the existing Go verifiers and derives
the merged decisions from their verified public statements, never planner output.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def validate_routes(routes, count):
    if set(routes) != {'certificate', 'full'} or count < 1:
        raise ValueError('expected two routing lists and a nonempty task')
    ids = []
    for branch in ['certificate', 'full']:
        part = routes[branch]
        if not isinstance(part, list) or any(type(i) is not int for i in part):
            raise ValueError('request IDs must be integer lists')
        if part != sorted(set(part)):
            raise ValueError('duplicate or unordered request IDs')
        ids += part
    if sorted(ids) != list(range(count)):
        raise ValueError('routes must cover every approved request exactly once')


def validate_pins(pin, task):
    cert, full = pin['certificate']['owner'], pin['full']
    if cert['original_root'] != full['original_root']:
        raise ValueError('branches authenticate different original catalogs')
    for field in ['n', 'd', 'k', 'input_bits', 'input_scale', 'centers']:
        if cert['model'][field] != full['model'][field]:
            raise ValueError('branch model mismatch: '+field)
    if full['model']['method'] != 'full' or pin['certificate']['version'] != 'group-compact-v3':
        raise ValueError('unsupported proof profiles')
    if task['comparison_base'] != full['model']['centers']:
        raise ValueError('unapproved reference centers')
    n, d, k = [full['model'][v] for v in ['n', 'd', 'k']]
    if not 1 <= n <= 1024 or not task['reuse_centers']:
        raise ValueError('invalid catalog or empty task')
    for centers in task['reuse_centers']:
        if len(centers) != k or any(len(c) != d or any(type(v) is not int for v in c) for c in centers):
            raise ValueError('invalid approved candidate shape')
    return n


def complete_queries(queries, n):
    if not queries or any(not q or any(type(i) is not int for i in q) for q in queries):
        raise ValueError('invalid public query batches')
    if [i for q in queries for i in q] != list(range(n)):
        raise ValueError('each branch must cover the complete ordered catalog')


def verify(root, pin_path, task_path, binary, invoke=None):
    root, binary = Path(root), Path(binary)
    pin, task = read(pin_path), read(task_path)
    if sha(binary) != pin['binary_sha256']:
        raise ValueError('verifier binary differs from independent pin')
    n = validate_pins(pin, task)
    routes = read(root/'routes.json')
    validate_routes(routes, len(task['reuse_centers']))
    # Reject accidentally exported private material before opening any proof.
    for p in root.rglob('*'):
        if p.is_file() and (p.suffix in ['.gob', '.pk', '.r1cs'] or p.name in ['input.json', 'group_private.json']):
            raise ValueError('private prover material in public bundle')
    decisions = [None]*len(task['reuse_centers'])

    def call(label, args):
        if invoke is not None:
            return invoke(label, [binary, *args])
        result = subprocess.run([str(binary), *map(str, args)], capture_output=True, text=True, timeout=600)
        (root/(label+'.log')).write_text(result.stdout+result.stderr, encoding='utf-8')
        if result.returncode:
            raise ValueError(label+' proof verification failed')

    # Derived tasks are rebuilt from the caller's approved task, not supplied by
    # the prover. Keeping them next to public exports permits independent replay.
    if routes['certificate']:
        cp = root/'certificate'
        # q is owner-pinned; canonical complete q-sized blocks are required.
        q = pin['certificate']['q']
        queries = [list(range(i, min(i+q, n))) for i in range(0, n, q)]
        complete_queries(queries, n)
        ct = dict(comparison_base=task['comparison_base'], queries=queries,
                  reuse_centers=[task['reuse_centers'][i] for i in routes['certificate']])
        save(root/'certificate_task.json', ct)
        save(root/'certificate_pin.json', pin['certificate'])
        common = ['-trust', root/'certificate_pin.json', '-public-task', root/'certificate_task.json']
        if (root/'refinement').exists():
            if 'refinement' not in pin:
                raise ValueError('refinement has no independent key pin')
            save(root/'refinement_pin.json', pin['refinement'])
            call('verify_refinement', ['-phase', 'refinement_verify', '-out', root/'refinement',
                 '-refinement-base', cp, '-refinement-trust', root/'refinement_pin.json', *common])
            verified = read(root/'refinement/refinement_verified_decisions.json')
        else:
            call('verify_certificate', ['-phase', 'group_verify', '-out', cp, *common])
            verified = read(cp/'group_verified_decisions.json')
        if len(verified) != len(routes['certificate']):
            raise ValueError('certificate result count differs from routing')
        for local, (i, record) in enumerate(zip(routes['certificate'], verified)):
            lo, hi, den = [int(record[v]) for v in ['lower', 'upper', 'denominator']]
            decision = 'worsening' if lo > 0 else 'improvement' if hi < 0 else 'equal' if lo == hi == 0 else 'inconclusive'
            if record['set'] != local or den <= 0 or lo > hi or decision == 'inconclusive' or record['decision'] != decision:
                raise ValueError('selected certificate did not verify a conclusive answer')
            decisions[i] = dict(request_id=i, path='certificate', lower=str(lo), upper=str(hi),
                                denominator=str(den), decision=decision)
    elif (root/'certificate').exists() or (root/'refinement').exists():
        raise ValueError('unrouted certificate evidence')
    if routes['full']:
        fp = root/'full'
        queries = read(root/'full_queries.json')
        complete_queries(queries, n)
        ft = dict(queries=queries, reuse_centers=[task['comparison_base']]+[task['reuse_centers'][i] for i in routes['full']])
        save(root/'full_task.json', ft)
        save(root/'full_pin.json', pin['full'])
        call('verify_full', ['-phase', 'verify', '-out', fp, '-trust', root/'full_pin.json', '-public-task', root/'full_task.json'])
        # Go verifies every sample's identity, centers, batch and nonnegative cost.
        costs = [0]*len(ft['reuse_centers'])
        for q in sorted({len(q) for q in queries}):
            phase = read(fp/f'full_reuse_{q}.json')
            for sample in phase['samples']:
                costs[sample.get('centre_set', 0)] += int(sample['psi'])
        for local, i in enumerate(routes['full'], 1):
            delta = costs[local]-costs[0]
            decisions[i] = dict(request_id=i, path='full', lower=str(delta), upper=str(delta),
                                denominator='1', decision='worsening' if delta>0 else 'improvement' if delta<0 else 'equal')
    elif (root/'full').exists():
        raise ValueError('unrouted Full evidence')
    if any(d is None for d in decisions):
        raise ValueError('unanswered request')
    result = dict(complete=True, decisions=decisions, original_root=pin['full']['original_root'],
                  approved_task_sha256=sha(task_path), independent_pin_sha256=sha(pin_path),
                  all_requests_verified=True, public_only=True)
    save(root/'verified.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'pin', 'task', 'binary']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(verify(a.root, a.pin, a.task, a.binary)))
