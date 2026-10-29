"""Check the unmeasured known-family exact alternative on frozen scale inputs."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code'))
from commitjl.family_affine import prepare, query


def main():
    path = ROOT/'tmp/paired_scale_20261022T055314Z/learned.json'
    owner = json.loads(path.read_text())
    rows, sets = owner['x'], owner['reuse_centers']
    frozen = prepare(rows, sets)
    checked = 0
    for centers in sets:
        result = query(frozen, centers)
        exact = [[sum((int(a)-int(b))**2 for a, b in zip(x, c)) for c in centers] for x in rows]
        assert result['labels'] == [min(range(len(c)), key=lambda j: c[j]) for c in exact]
        assert result['costs'] == [min(c) for c in exact]
        checked += len(rows)
    # New out-of-family center must not silently use unapproved scores.
    novel = [[10**20]*len(rows[0]), sets[0][0]]
    try:
        query(frozen, novel)
    except ValueError:
        pass
    else:
        raise AssertionError('unknown center accepted')
    # Arbitrary-size integers and first-index ties: no float rounding or overflow.
    huge = 10**30
    fixture = prepare([[huge, -huge]], [[[huge+1, -huge], [huge-1, -huge]]])
    assert query(fixture, [[huge+1, -huge], [huge-1, -huge]]) == dict(labels=[0], costs=[1], total=1)
    out = ROOT/'paper_results/review_controls/referee_20261023'
    out.mkdir(parents=True, exist_ok=True)
    report = dict(input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        n=len(rows), center_sets=len(sets), unique_centers=len(frozen['centers']),
        exact_row_set_comparisons=checked, all_labels_and_costs_match=True,
        stored_scalars_per_row=frozen['scalar_count_per_row'],
        nonconstant_scalars_per_row=frozen['nonconstant_scalar_count_per_row'],
        compressed_head_energy_scalars_per_row=len(owner['model']['basis'])+1,
        security_scope='Native exact-integer reference only; no authenticated family circuit or proof timing measured.',
        assumptions='Whole center family approved before statistics are derived; unknown out-of-family centers rejected.')
    (out/'known_family_affine.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
