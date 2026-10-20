"""Record honest controls and classified circuit rejections before server retirement."""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
from commitjl.zk_bridge import build_sparse_instance, run_prove


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    out = Path(args.root)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cases = [('honest', 'groth16'), ('honest', 'plonk')]
    cases += [(name, 'groth16') for name in
              ['wrong_A', 'wrong_X_open', 'wrong_assign', 'tie_break', 'wrong_psi', 'wraparound']]
    # Extra isolated cases: ensure a real nearest-centroid tie and coherent
    # objective/commitments for an out-of-range input, rather than a stale cost.
    cases += [('isolated_tie_break', 'groth16'), ('isolated_input_range', 'groth16')]
    rows = []
    for name, backend in cases:
        rng = np.random.default_rng(42)
        x = rng.integers(-50, 50, size=(20, 32)).astype(np.int64)
        centers = rng.integers(-50, 50, size=(3, 32)).astype(np.int64)
        tamper = None if name == 'honest' or name.startswith('isolated_') else name
        if name == 'isolated_tie_break':
            centers[0] = x[0]
            centers[1] = x[0]
        if name == 'isolated_input_range':
            x[0, 0] = 4097  # InputCoordinateBound is 4096 in circuit.go.
        inst = build_sparse_instance(x, centers, seed=b'commitjl-test-zk-seed',
                                     q=2, m=8, backend=backend, range_bits=48, tamper=tamper)
        if name == 'isolated_tie_break':
            assert inst.assign[0] == 0
            inst.assign[0] = 1  # Equal cost, wrong deterministic winner.
        (out / f'{name}_{backend}_instance.json').write_text(json.dumps(inst.to_json_dict()))
        begin = time.time()
        result = run_prove(inst, timeout=300, max_rss_bytes=12 * 1024**3,
                           build_if_missing=False)
        if name == 'honest':
            passed = result.ok and result.verify_ok
            classification = 'valid_proof' if passed else 'unexpected_honest_failure'
        else:
            # gnark unsatisfied-constraint errors; never treat infrastructure
            # errors, compiler failures, OOM, or timeouts as successful rejection.
            circuit = (result.failure_kind == 'prove_or_verify_failed'
                       and result.error.startswith('prove:')
                       and any(s in result.error.lower() for s in
                               ('constraint', 'unsatisfied', 'assertion', 'solver')))
            preflight = (name == 'wrong_A'
                         and result.failure_kind == 'prove_or_verify_failed'
                         and 'projection' in result.error.lower()
                         and 'mismatch' in result.error.lower())
            passed = not result.ok and not result.verify_ok and (circuit or preflight)
            classification = 'circuit_rejection' if circuit else (
                'host_preflight_rejection' if preflight else 'unexpected_failure_or_acceptance')
        row = dict(case=name, backend=backend, passed=bool(passed),
                   classification=classification, wall_seconds=time.time()-begin,
                   result=dataclasses.asdict(result))
        rows.append(row)
        (out / f'{name}_{backend}_result.json').write_text(json.dumps(row, indent=2))
        summary = dict(status='running', started_epoch=started, cases=rows)
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps({k: row[k] for k in ['case', 'backend', 'passed', 'classification', 'wall_seconds']}), flush=True)
    summary.update(status='completed', passed=all(r['passed'] for r in rows),
                   finished_epoch=time.time(), wall_seconds=time.time()-started)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary['passed'] else 1)


if __name__ == '__main__':
    main()
