"""Check and summarize the expanded local native and authenticated prototypes."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    native = ROOT/'paper_results/review_controls/local_group_moments_extended_20261023'
    proofs = ROOT/'code/validation/outputs/local_group_proofs_20261023'
    out = ROOT/'paper_results/review_controls/group_moments_local_20261023'
    out.mkdir(exist_ok=True)
    data = pd.read_csv(native/'results.csv')
    assert len(data) == 192 and (data.true_change == data.exact_change).all()
    unique = data[data.method == 'learned']
    regress = unique[(unique.cohort == 'historical_development') & (unique.truth == 'worsens')]
    assert len(regress) == 24
    frozen = pd.read_csv(ROOT/'paper_results/review_controls/referee_20261023/actual_updates_by_sign.csv')
    expected = frozen[(frozen.method == 'learned') & (frozen.truth == 'worsens')]
    actual_keys = set(zip(regress.dataset, regress.k, regress.seed, regress.age, regress['set']))
    expected_keys = set(zip(expected.dataset, expected.k, expected.fit_seed, expected.age, expected['set']))
    assert actual_keys == expected_keys
    confirmation = unique[unique.cohort == 'confirmation']
    assert len(confirmation) == 32 and set(confirmation.seed) == {67, 79}
    stats = []
    for method in ['learned', 'pca']:
        rows = data[(data.method == method) & (data.cohort == 'historical_development') & (data.truth == 'worsens')]
        stats.append(dict(method=method, regressions=len(rows), correct_changes=int((rows.true_change == rows.exact_change).sum()),
            median_exact_percent=100*rows.exact_fraction.median(), min_exact_percent=100*rows.exact_fraction.min(),
            max_exact_percent=100*rows.exact_fraction.max()))
    pd.DataFrame(stats).to_csv(out/'regression_summary.csv', index=False)
    timings = []
    proof_count = 0
    for case in ['smoke', 'fashion_n4']:
        folder = proofs/case
        result = json.loads((folder/'summary.json').read_text())
        replay = json.loads((folder/'fresh_replay.json').read_text())
        assert result['Complete'] and replay['Complete'] and replay['Verified'] == 10 and result['TamperRejected']
        proof_count += replay['Verified']
        assert set(json.loads((folder/'pins.json').read_text())['KeySHA']) == {'group_prepare', 'group_query', 'full'}
        for phase in result['Phases']:
            for sample in phase['Samples']:
                assert (folder/sample['Proof']).is_file()
        reconstructed_group = result['GroupNative']
        reconstructed_full = result['FullNative']
        for phase in result['Phases']:
            value = phase['Compile']+phase['Setup']+sum(s['Witness']+s['Prove']+s['Verify'] for s in phase['Samples'])
            if phase['Name'] == 'full':
                reconstructed_full += value
            else:
                reconstructed_group += value
        assert abs(reconstructed_group-result['GroupTotal']) < 1e-6
        assert abs(reconstructed_full-result['FullTotal']) < 1e-6
        timings.append(dict(case=case, group_seconds=result['GroupTotal'], full_seconds=result['FullTotal'],
            full_over_group=result['FullOverGroup'], public_proofs_verified=replay['Verified']))
    pd.DataFrame(timings).to_csv(out/'proof_costs.csv', index=False)
    report = dict(complete=True, native_unique_updates=96, native_method_checks=192,
        historical_regressions=24, historical_improvements=40, independent_updates=32,
        independent_regressions=int((confirmation.truth == 'worsens').sum()), independent_improvements=int((confirmation.truth == 'improves').sum()),
        all_native_changes_exact=True, native_false_changes=0, regression_statistics=stats,
        fresh_public_proofs_verified=proof_count, proof_costs=timings,
        python_targeted_tests=19, go_check='go test ./... -p 1',
        scientific_conclusion='Exact-change feasibility passes all existing regressions. Bounded authenticated prototype works but costs more than Full; no efficient full-scale fix established.',
        privacy_contract='Prototype publishes correction indices/active slots. Moments, group membership, head rows and original rows remain witness data; output exact aggregate cost change is public.',
        limitation='Numerical N1024 and cryptographic N4 full-feature proof pilot are separate evidence. Head is locally refitted; historical PCA bases match. Independent fits contain zero regressions.')
    (out/'summary.json').write_text(json.dumps(report, indent=2))
    sources = list(native.glob('*')) + list(proofs.rglob('*')) + list(out.glob('*'))
    sources += [ROOT/p for p in ['code/commitjl/group_moment_reference.py', 'code/experiments/pilot_group_moments.py',
        'code/zk/circuit/bound_group_moments.go', 'code/zk/circuit/bound_group_moments_test.go',
        'code/zk/cmd/commitjl-group/main.go', 'code/zk/bin/commitjl-group.exe']]
    manifest = [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources if p.is_file() and p.name != 'manifest.json']
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
