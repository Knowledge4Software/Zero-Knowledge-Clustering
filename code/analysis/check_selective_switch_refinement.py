"""Native gate for bounded exact-row refinement of the selected group method.

Width-only ordering and fixed caps are chosen before examining exact row
corrections. This is development evidence, not proof-cost measurement.
"""
from pathlib import Path
import argparse, json, time
import numpy as np
from threadpoolctl import threadpool_limits
from campaigns.local_fast_tradeoff import save, sha
from commitjl.group_change_bounds import exact_old_correction_bounds
from commitjl.group_moment_reference import prepare_groups
from commitjl.jl_certificate_pilots import hint_order, refinement_work

REPO = Path(__file__).resolve().parents[2]


def distances(rows, centers):
    out = np.empty((len(rows), len(centers)), dtype=np.int64)
    for start in range(0, len(rows), 32):
        out[start:start+32] = ((rows[start:start+32, None] - centers[None])**2).sum(2)
    return out


def run(source, out):
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    records, pins = [], {}
    caps = [0, 1, 2, 4, 8, 16]
    save(out/'protocol.json', dict(caps=caps, ordering='Descending certified row width; row index breaks ties',
        source=str(source), runner_sha256=sha(Path(__file__)), native_only=True,
        selection='All distinct nonidentity updates from all stored fresh fits, both representations, N128 and N1024. Previously observed development grid; not independent new fits.',
        claim='No proof speed or authenticated refinement implementation is inferred.'))
    fits = json.loads((source/'fit_records.json').read_text())
    with threadpool_limits(limits=1):
        for fit in fits:
            path = source/'fits'/f"{fit['fit']}_{fit['representation']}.npz"
            assert sha(path) == fit['archive_sha256']
            pins[path.relative_to(REPO).as_posix()] = sha(path)
            data = np.load(path)
            for n in [128, 1024]:
                rows = data['audit'][:n].astype(np.int64)
                centers = data['centers'].astype(np.int64)
                old = centers[0]
                model = dict(method='learned_energy', n=n, d=rows.shape[1], k=len(old),
                    input_bits=12, basis_scale=16384, basis=data['auxiliary_basis'].tolist(), centers=old.tolist(),
                    tail_width=0, tail_rows=[], tail_signs=[])
                od = distances(rows, old)
                groups = od.argmin(1)
                prepared = prepare_groups(rows, groups.tolist(), len(old))
                seen = {old.tobytes()}
                for index, new in enumerate(centers[1:], 1):
                    if new.tobytes() in seen:
                        continue
                    seen.add(new.tobytes())
                    nd = distances(rows, new)
                    truth = int(nd.min(1).sum() - od.min(1).sum())
                    b = exact_old_correction_bounds(model, rows, prepared, old, new)
                    den = b['cost_denominator']
                    exact = [int(v)*den for v in nd.min(1)-nd[np.arange(n), groups]]
                    order = hint_order(b, None, 'width')
                    for cap in caps:
                        r = refinement_work(b, exact, order, cap)
                        assert r['lower'] <= truth*den <= r['upper']
                        expected = 'improves' if truth < 0 else ('worsens' if truth > 0 else 'equal')
                        assert r['decision'] in ['inconclusive', expected]
                        records.append(dict(fit=fit['fit'], representation=fit['representation'], n=n,
                            source_set=index, truth=truth, strict_switch_rows=int((nd.min(1)<nd[np.arange(n), groups]).sum()),
                            cap=cap, selected=order[:r['exact_rows']], **r))
                save(out/'records.json', records)
            print(json.dumps(dict(fit=fit['fit'], representation=fit['representation'], records=len(records))), flush=True)
    summary = []
    for n in [128, 1024]:
        for representation in ['ordered_head', 'pca']:
            for cap in caps:
                r = [v for v in records if v['n']==n and v['representation']==representation and v['cap']==cap]
                summary.append(dict(n=n, representation=representation, cap=cap, updates=len(r),
                    conclusive=sum(v['decision']!='inconclusive' for v in r),
                    regressions=sum(v['truth']>0 for v in r),
                    detected_regressions=sum(v['truth']>0 and v['decision']=='worsens' for v in r),
                    total_exact_rows=sum(v['exact_rows'] for v in r),
                    maximum_exact_rows=max([v['exact_rows'] for v in r], default=0)))
    result = dict(complete=True, native_only=True, false_conclusive=0, all_intervals_contain_truth=True,
        elapsed_seconds=time.monotonic()-started, sources=pins, results=summary)
    save(out/'summary.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=REPO/'paper_results/review_controls/rank79_independent_replication_20261025')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    run(a.source.resolve(), a.out.resolve())
