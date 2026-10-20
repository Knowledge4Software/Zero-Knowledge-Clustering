"""Read-only, offline audit of preserved research inputs and their local sources.

Run from code/: python -u -m tools.validate_research_datasets --out results/dataset_validation_20261011
The audit writes only to its output folder; archived results and caches are not changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import gc
import gzip
import hashlib
import io
import json
from pathlib import Path
import socket
import time

import numpy as np
from threadpoolctl import threadpool_limits

from commitjl.metrics import sample_pair_set
from data.loaders import gaussian_mixture, dbscan_margin_dataset
from experiments.rq1_complete import DATASETS

CODE = Path(__file__).resolve().parents[1]
CACHE = CODE / '.cache/commitjl_data'
BASE = CODE / 'results/server_retirement_20261007/compact_arrays'
RECENT = CODE / 'results/laptop_comparison_20261011'


def array_hash(a):
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@contextmanager
def offline():
    original_connect = socket.socket.connect
    original_create = socket.create_connection
    def forbidden(*args, **kwargs):
        raise RuntimeError('Network access forbidden by dataset audit')
    socket.socket.connect = forbidden
    socket.create_connection = forbidden
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.create_connection = original_create


class Preserved:
    def __init__(self):
        self.index = json.loads((BASE / 'array_index.json').read_text())['files']
        self.verified = set()

    def get(self, path, key):
        entry = self.index[path]['arrays'][key]
        obj = entry['object']
        with gzip.open(BASE / 'objects' / (obj + '.npy.gz'), 'rb') as stream:
            payload = stream.read()
        if hashlib.sha256(payload).hexdigest() != obj:
            raise ValueError(f'Archive payload hash mismatch: {path}:{key}')
        a = np.load(io.BytesIO(payload), allow_pickle=False)
        if list(a.shape) != entry['shape'] or str(a.dtype) != entry['dtype']:
            raise ValueError(f'Archive shape/dtype mismatch: {path}:{key}')
        self.verified.add(obj)
        return a

    def load(self, path):
        return {key: self.get(path, key) for key in self.index[path]['arrays']}


def check(record, name, okay, **details):
    record.setdefault('checks', []).append(dict(name=name, passed=bool(okay), **details))


def compare_arrays(a, b):
    if a.shape != b.shape:
        return dict(shape_equal=False, exact=False, numerically_equal=False)
    maximum = 0.
    changed = 0
    close = True
    for first in range(0, len(a), 64):
        left, right = a[first:first+64], b[first:first+64]
        maximum = max(maximum, float(np.max(np.abs(left-right), initial=0)))
        changed += int(np.count_nonzero(left != right))
        close &= bool(np.allclose(left, right, rtol=1e-10, atol=1e-12))
    return dict(shape_equal=True, exact=changed == 0, numerically_equal=close,
                changed_elements=changed, maximum_absolute_difference=maximum)


def statistics(x, y):
    row_groups = defaultdict(list)
    for i, row in enumerate(x):
        row_groups[array_hash(row)].append(i)
    duplicates = [indices for indices in row_groups.values() if len(indices) > 1]
    zero = np.flatnonzero(np.all(x == 0, axis=1))
    constant = np.flatnonzero(x.min(axis=0) == x.max(axis=0))
    norms = np.sqrt(np.einsum('ij,ij->i', x, x))
    return dict(n=len(x), d=x.shape[1], dtype=str(x.dtype), sha256=array_hash(x),
                finite=bool(np.isfinite(x).all()), minimum=float(x.min()), maximum=float(x.max()),
                zero_rows=len(zero), zero_row_indices=zero.tolist(),
                constant_columns=len(constant), constant_column_indices=constant.tolist(),
                all_zero_columns=int(np.count_nonzero(np.all(x == 0, axis=0))),
                nonzero_fraction=float(np.count_nonzero(x)/x.size),
                duplicate_extra_rows=sum(len(g)-1 for g in duplicates), duplicate_groups=duplicates,
                duplicate_groups_with_conflicting_labels=sum(len(np.unique(y[g])) > 1 for g in duplicates) if len(y) else 0,
                row_norm_min=float(norms.min()), row_norm_max=float(norms.max()),
                class_counts={str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))} if len(y) else {},
                label_sha256=array_hash(y) if len(y) else None)


def chosen_indices(source_n, n, seed):
    return np.random.default_rng(seed).choice(source_n, n, replace=False) if n < source_n else np.arange(source_n)


def scan_numeric(path, indices, d, expected_n, delimiter=None, arff=False):
    """Scan every source row, retaining only the exact ordered research subset."""
    selected = {int(row): pos for pos, row in enumerate(indices)}
    x = np.empty((len(indices), d), dtype=np.float64)
    labels = np.empty(len(indices), dtype=np.int64) if arff else None
    count = 0
    low, high = np.inf, -np.inf
    integral = True
    counts = Counter()
    md5 = hashlib.md5()
    ready = not arff
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as stream:
        for raw_line in stream:
            md5.update(raw_line)
            line = raw_line.strip()
            if not ready:
                ready = line.lower() == b'@data'
                continue
            if not line or line.startswith(b'%'):
                continue
            values = np.fromstring(line.decode('ascii'), sep=delimiter or ' ')
            if len(values) != d + int(arff) or not np.isfinite(values).all():
                raise ValueError(f'Malformed source row {count}: {path}')
            features = values[:d]
            low = min(low, float(features.min()))
            high = max(high, float(features.max()))
            integral &= bool(np.all(features == np.rint(features)))
            if arff:
                target = int(values[-1])
                if target != values[-1] or not 0 <= target <= 9:
                    raise ValueError(f'Invalid image class: {path}:{count}')
                counts[target] += 1
            if count in selected:
                pos = selected[count]
                x[pos] = features
                if arff:
                    labels[pos] = target
            count += 1
    if count != expected_n:
        raise ValueError(f'Wrong source row count: {path}: {count} != {expected_n}')
    return x, labels, dict(rows=count, d=d, minimum=low, maximum=high,
                           integral=integral, full_class_counts=dict(counts), uncompressed_md5=md5.hexdigest())


def reconstruct(name, record, out):
    manifest = record['manifest']
    meta = manifest['dataset']
    kwargs = manifest['loader_kwargs']
    seed = kwargs['seed']
    n = meta['n']
    if meta['source'] == 'synthetic':
        loader = DATASETS[name][0]
        generated = (gaussian_mixture if loader == 'gaussian_mixture' else dbscan_margin_dataset)(**kwargs)
        return generated['X'], generated.get('y', np.array([])), dict(type='deterministic generator', seed=seed)
    indices = chosen_indices(meta['source_n'], n, seed)
    np.save(out / f'{name}_source_indices.npy', indices, allow_pickle=False)
    if name in ('mnist', 'fashion_mnist'):
        data_id = 554 if name == 'mnist' else 40996
        folder = CACHE / 'openml/openml.org'
        with gzip.open(folder / f'api/v1/json/data/{data_id}.gz', 'rt') as stream:
            description = json.load(stream)['data_set_description']
        candidates = list((folder / 'data/v1/download' / description['file_id']).glob('*.arff.gz'))
        if len(candidates) != 1:
            raise ValueError('Missing or ambiguous image raw cache')
        source = candidates[0]
        x, y, info = scan_numeric(source, indices, 784, 70000, delimiter=',', arff=True)
        check(record, 'image source range and integer pixels', info['integral'] and info['minimum'] >= 0 and info['maximum'] <= 255)
        check(record, 'OpenML file checksum equals locally preserved metadata', info['uncompressed_md5'] == description['md5_checksum'])
        info.update(type='cached OpenML ARFF', data_id=data_id, version=description['version'],
                    name=description['name'], declared_md5=description['md5_checksum'], raw_sha256=file_hash(source))
        return x, y, info
    if name in ('gisette', 'arcene'):
        folder = CACHE / ('gisette' if name == 'gisette' else 'arcene/ARCENE')
        source = folder / f'{name}_train.data'
        x, _, info = scan_numeric(source, indices, meta['source_d'], meta['source_n'])
        y_full = np.loadtxt(folder / f'{name}_train.labels', dtype=np.int64)
        check(record, 'UCI label count and domain', y_full.shape == (meta['source_n'],) and set(np.unique(y_full)) == {-1, 1})
        info.update(type='cached UCI training text', raw_sha256=file_hash(source),
                    labels_sha256=file_hash(folder / f'{name}_train.labels'),
                    full_class_counts={str(k): int(v) for k, v in zip(*np.unique(y_full, return_counts=True))})
        return x, y_full[indices], info
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    raw = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'),
                            data_home=str(CACHE), download_if_missing=False)
    if len(raw.data) != 11314 or len(raw.target_names) != 20:
        raise ValueError('Unexpected 20NG train corpus')
    vectorizer = TfidfVectorizer(max_features=kwargs['max_features'], stop_words='english')
    sparse = vectorizer.fit_transform(raw.data)
    names = vectorizer.get_feature_names_out().tolist()
    (out / '20ng_reconstructed_feature_names.json').write_text(json.dumps(names, indent=2), encoding='utf-8')
    info = dict(type='cached sklearn 20NG train + recomputed TF-IDF', rows=len(raw.data), d=sparse.shape[1],
                raw_sha256=file_hash(CACHE / '20news-bydate_py3.pkz'),
                remove=['headers', 'footers', 'quotes'], max_features=kwargs['max_features'],
                stop_words='english', norm='l2', fit_scope='all 11314 training texts before subset selection',
                zero_vector_source_rows=int(np.count_nonzero(np.diff(sparse.indptr) == 0)),
                selected_empty_cleaned_texts=int(sum(not raw.data[i].strip() for i in indices)),
                selected_nonempty_texts_with_zero_features=int(sum(bool(raw.data[i].strip()) and sparse.indptr[i] == sparse.indptr[i+1] for i in indices)),
                target_names=raw.target_names)
    return sparse[indices].toarray(), raw.target[indices].astype(np.int64), info


def audit_baselines(archive, name, x, pairs, record):
    prefix = f'results/paper_server_20261006/rq1/{name}/artifacts/'
    pca = archive.load(prefix + 'pca.npz')
    components, mean, variance = pca['components'], pca['mean'], pca['explained_variance']
    orth_error = float(np.max(np.abs(components @ components.T - np.eye(len(components)))))
    check(record, 'PCA dimensions, centered rank and finite arrays', components.shape[1] == x.shape[1] and len(components) <= min(len(x)-1, x.shape[1]) and mean.shape == (x.shape[1],) and variance.shape == (len(components),) and all(np.isfinite(v).all() for v in pca.values()))
    check(record, 'PCA component orthonormality', orth_error < 1e-8, maximum_error=orth_error)
    check(record, 'PCA mean equals preserved dataset mean', np.allclose(mean, x.mean(axis=0), rtol=1e-12, atol=1e-10))
    check(record, 'PCA nonnegative ordered variances', bool(np.all(variance >= 0) and np.all(np.diff(variance) <= 1e-8 * max(1., variance.max()))))
    projection = x @ components[:min(64, len(components))].T
    measured_variance = projection.var(axis=0, ddof=1)
    var_error = float(np.max(np.abs(measured_variance-variance[:projection.shape[1]]) / np.maximum(variance[:projection.shape[1]], 1e-30)))
    record['pca'] = dict(rank=len(components), orthonormality_max_error=orth_error,
                         variance_check_first_components=projection.shape[1], actual_vs_randomized_saved_variance_max_relative_error=var_error)
    # Randomized PCA can understate actual projected variance; report the approximation.
    if var_error > .01:
        record['warnings'].append('Saved randomized PCA variance differs from actual projected variance by over 1% in the first 64 components.')
    record['baselines'] = []
    xx = np.einsum('ij,ij->i', x, x)
    sample = np.linspace(0, len(x)-1, min(32, len(x)), dtype=int)
    original_pairs = np.sum((x[pairs[:,0]] - x[pairs[:,1]])**2, axis=1)
    record['statistics']['sampled_zero_distance_pairs'] = int(np.count_nonzero(original_pairs == 0))
    recent_inputs = RECENT / 'jobs' / f'geometry_{name}' / 'inputs.npz'
    with np.load(recent_inputs, allow_pickle=False) as saved:
        check(record, 'Recent campaign uses exact pair indices', np.array_equal(saved['pairs'], pairs))
        check(record, 'Recent campaign original pair distances', np.allclose(saved['original_pair_sq'], original_pairs, rtol=1e-12, atol=1e-10))
    precision_root = RECENT / 'jobs' / f'precision_{name}'
    precision_n = min(128, len(x))
    precision_pairs = np.column_stack((np.arange(precision_n-1), np.arange(1, precision_n)))
    with np.load(precision_root / 'inputs.npz', allow_pickle=False) as saved:
        check(record, 'Precision campaign prefix pair indices and distances',
              np.array_equal(saved['pairs'], precision_pairs) and
              np.allclose(saved['original_pair_sq'], np.sum((x[:precision_n-1]-x[1:precision_n])**2, axis=1), rtol=1e-12, atol=1e-10))
    for k in (5, 10):
        for start in range(5):
            full = archive.load(prefix + f'full_k{k}_start{start}.npz')
            c, labels, saved_sq = full['centroids'], full['labels'], full['squared_distances']
            sq = np.maximum(xx[:,None] + np.einsum('ij,ij->i', c, c)[None,:] - 2*x@c.T, 0)
            direct = np.sum((x[sample,None,:] - c[None,:,:])**2, axis=2)
            valid = c.shape == (k, x.shape[1]) and labels.shape == (len(x),) and saved_sq.shape == (len(x), k)
            check(record, f'Full k{k} start{start} distances and labels', valid and np.isfinite(c).all() and np.isfinite(saved_sq).all() and np.array_equal(labels, saved_sq.argmin(axis=1)) and np.array_equal(labels, sq.argmin(axis=1)) and np.allclose(saved_sq, sq, rtol=1e-10, atol=1e-8) and np.allclose(saved_sq[sample], direct, rtol=1e-12, atol=1e-8))
            record['baselines'].append(dict(k=k, start=start, max_distance_difference=float(np.abs(saved_sq-sq).max()),
                                             empty_clusters=int(np.count_nonzero(np.bincount(labels, minlength=k) == 0))))
            if start in (3, 4):
                with np.load(RECENT / 'jobs' / f'geometry_{name}' / f'centers_k{k}_start{start}.npz', allow_pickle=False) as saved:
                    u = saved['basis']
                    delta = c[1:] - c[0]
                    relative_residual = float(np.linalg.norm(delta - (delta@u)@u.T)/max(np.linalg.norm(delta), 1e-30))
                    check(record, f'Recent k{k} start{start} centers, labels, protected span', np.array_equal(saved['centroids'], c) and np.array_equal(saved['original_labels'], labels) and np.allclose(saved['original_centroid_sq'], saved_sq, rtol=1e-12, atol=1e-8) and np.allclose(u.T@u, np.eye(u.shape[1]), rtol=1e-10, atol=1e-10) and relative_residual < 1e-10, protected_rank=u.shape[1], relative_span_residual=relative_residual)
            if start == 3:
                factor = 4096 / max(float(np.max(np.abs(x))), float(np.max(np.abs(c))), 1e-15)
                with np.load(precision_root / f'centers_k{k}.npz', allow_pickle=False) as saved:
                    check(record, f'Precision k{k} preserved raw centers, prefix X quantization and distances',
                          np.array_equal(saved['centroids'], c) and
                          np.array_equal(saved['quantized_X'], np.rint(x[:precision_n]*factor).astype(np.int64)) and
                          np.array_equal(saved['quantized_centroids'], np.rint(c*factor).astype(np.int64)) and
                          np.allclose(saved['original_centroid_sq'], saved_sq[:precision_n], rtol=1e-12, atol=1e-8))


def write_report(out, result):
    lines = ['# Research dataset validation', '',
             'Offline, read-only validation of the eight preserved paper subsets used in the latest Full/PCA/protected-JL comparisons. Local raw caches are independently parsed or re-vectorized; no archived inputs were changed.', '',
             '| Dataset | Preserved n × d | Source rows | Classes | Zero rows | Duplicate extra rows | Constant features | Status |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for name, record in result['datasets'].items():
        s = record.get('statistics', {})
        meta = record['manifest']['dataset']
        lines.append(f"| {name} | {meta['n']} × {meta['d']} | {meta.get('source_n', meta['n'])} | {len(s.get('class_counts', {})) or 'unlabelled'} | {s.get('zero_rows', '?')} | {s.get('duplicate_extra_rows', '?')} | {s.get('constant_columns', '?')} | {record.get('status', 'pending')} |")
    lines += ['', '## Findings and interpretation', '']
    for name, record in result['datasets'].items():
        comparison = record.get('source_comparison', {})
        if comparison:
            lines.append(f"- **{name}:** raw-source reproduction {'is byte-exact' if comparison.get('exact') else 'matches numerically within rtol=1e-10, atol=1e-12' if comparison.get('numerically_equal') else 'FAILED'}. Maximum absolute difference: {comparison.get('maximum_absolute_difference', 'n/a')}; changed entries: {comparison.get('changed_elements', 'n/a')}.")
        for warning in record['warnings']:
            lines.append(f'  - {warning}')
        for c in record.get('checks', []):
            if not c['passed']:
                lines.append(f"  - **FAILED CHECK:** {c['name']}")
        if 'error' in record:
            lines.append(f"  - **ERROR:** {record['error']}")
    lines += ['', '## Scope of the checks', '',
              '- Content-addressed array payload hashes, original server X/label hashes, shapes/dtypes, finite values, source/subset reconstruction and ordering, label domains/counts, row duplicates, zero rows, and constant columns.',
              '- All 80 saved Full baseline artifacts (k=5/10, starts 0–4): centroids, every saved point-center distance, nearest-center labels, and independent direct-distance checks on 32 points per artifact.',
              '- All eight cached PCA matrices: rank, finite values, mean, component orthonormality, variance order, and reported first-64-component variance approximation error.',
              '- Latest campaign inputs: exact pair indices, original pair distances, all 32 geometry center/basis snapshots, every geometry row dataset hash, and the precision prefix inputs/centers/quantization factors (precision rows do not store a dataset hash).',
              '- Remaining preserved dataset.npz containers: unique arrays are hash-checked and checked for finite values and label/pair dimensions; this does not re-run every older experiment or identify every synthetic generator.', '',
              '## Limits and recommendations', '',
              '- Local source/metadata consistency is verified. UCI files were not re-downloaded and their publisher authenticity was not independently re-attested. OpenML raw ARFF checksums are compared against locally preserved OpenML metadata.',
              '- These are fixed-centroid, within-sample fidelity benchmarks. Full ARI=1 is self-agreement, not ground-truth accuracy; supervised accuracy and out-of-sample generalization are separate questions.',
              '- MNIST/Fashion subsets are random samples from the combined 70,000-row OpenML corpus. Gisette/Arcene are training data, and 20NG uses its 11,314-text training corpus. There is no independent evaluation split in these comparisons.',
              '- TF-IDF is fit on the entire training corpus before choosing 3,000 rows. Zero vectors after cleaning/vocabulary restriction and their duplicate rows are retained in the original experiment; zero-distance pairs must be reported and excluded from relative-error denominators.',
              '- Raw image/UCI feature magnitudes are retained (no standardization); TF-IDF is L2-normalized. All methods receive the same X. Changing preprocessing changes the research question and requires new paired runs.',
              '- Cached randomized PCA is an approximation, not a guarantee of the optimal low-rank PCA basis. Small close comparisons should be confirmed with stronger PCA fitting; it is not evidence of invalid source data.',
              '- Repeated experiments on the same subset are exploratory. Choose configurations first, then confirm them on new map seeds and an independent sample before claiming general superiority.', '',
              f"Complete: {result.get('complete')}. Passed checks: {result.get('passed_checks')}. Failed checks: {result.get('failed_checks')}. Wall time: {result.get('wall_seconds', 0):.1f} seconds.", '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    (out / 'validation.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a new output directory')
    args.out.mkdir(parents=True)
    start_time = time.perf_counter()
    import sklearn, scipy
    result = dict(complete=False, network='blocked during audit', datasets={},
                  environment=dict(numpy=np.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__, threads=4),
                  provenance=dict(audit_source_sha256=file_hash(Path(__file__)), loader_source_sha256=file_hash(CODE/'data/loaders.py'), archive_index_sha256=file_hash(BASE/'array_index.json')))
    archive = Preserved()
    known_objects = {}
    with offline(), threadpool_limits(limits=4):
        for name in DATASETS:
            record = result['datasets'][name] = dict(warnings=[])
            record['manifest'] = json.loads((CODE/f'results/server_paper_20261006/rq1/{name}/manifest.json').read_text())
            prefix = f'results/paper_server_20261006/rq1/{name}/artifacts/'
            try:
                data = archive.load(prefix + 'dataset.npz')
                x, y, pairs = data['X'], data['y'], data['pairs']
                s = record['statistics'] = statistics(x, y)
                meta = record['manifest']['dataset']
                check(record, 'Original matrix hash and dimensions', s['sha256'] == meta['sha256'] and x.shape == (meta['n'], meta['d']))
                check(record, 'Finite float64 matrix', s['finite'] and x.dtype == np.float64)
                check(record, 'Label dimensions and original label hash', (not len(y) and meta.get('labels_sha256') is None) or (y.shape == (len(x),) and s['label_sha256'] == meta.get('labels_sha256', s['label_sha256'])))
                expected_classes = {'mnist': 10, 'fashion_mnist': 10, 'gisette': 2, 'arcene': 2, '20ng_tfidf': 20, 'gaussian_easy': 5, 'gaussian_hard': 8, 'dbscan_margin': 0}[name]
                check(record, 'Expected class count', len(s['class_counts']) == expected_classes)
                check(record, 'Valid unique unordered index pairs', pairs.shape == (2000, 2) and pairs.dtype.kind in 'iu' and np.all(pairs[:,0] < pairs[:,1]) and np.all(pairs >= 0) and np.all(pairs < len(x)) and len(np.unique(pairs, axis=0)) == len(pairs))
                check(record, 'Reproducible saved pair set', np.array_equal(pairs, np.asarray(sample_pair_set(len(x), 2000, record['manifest']['pair_seed']))))
                source_x, source_y, source_info = reconstruct(name, record, args.out)
                record['raw_source'] = source_info
                comparison = record['source_comparison'] = compare_arrays(x, source_x)
                check(record, 'Independent source/subset reconstruction', comparison['numerically_equal'])
                check(record, 'Independent raw label alignment', np.array_equal(y, source_y))
                del source_x, source_y
                gc.collect()
                if s['zero_rows']:
                    record['warnings'].append(f"{s['zero_rows']} zero feature vectors; retained by the original benchmark.")
                if s['duplicate_extra_rows']:
                    record['warnings'].append(f"{s['duplicate_extra_rows']} duplicate extra rows; {s['duplicate_groups_with_conflicting_labels']} duplicate groups have different true labels.")
                if s['constant_columns']:
                    record['warnings'].append(f"{s['constant_columns']} constant feature columns are retained; nominal d includes them.")
                audit_baselines(archive, name, x, pairs, record)
                if name in ('mnist', 'fashion_mnist'):
                    np.savez_compressed(args.out/f'{name}_preview.npz', images=np.stack([x[np.flatnonzero(y == label)[0]].reshape(28, 28) for label in range(10)]))
                known_objects[archive.index[prefix+'dataset.npz']['arrays']['X']['object']] = name
                record['status'] = 'passed' if all(c['passed'] for c in record['checks']) else 'failed'
                del data, x, y, pairs
            except Exception as exc:
                record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            gc.collect()
            result['wall_seconds'] = time.perf_counter()-start_time
            write_report(args.out, result)
            print(json.dumps(dict(dataset=name, status=record['status'], seconds=round(result['wall_seconds'], 1), warnings=record['warnings'], error=record.get('error'))), flush=True)
        # Check all older preserved dataset containers once per content-addressed object.
        result['other_dataset_containers'] = []
        for path, entry in archive.index.items():
            if not path.endswith('/dataset.npz'):
                continue
            item = dict(path=path, shape=entry['arrays']['X']['shape'], matches_primary_dataset=known_objects.get(entry['arrays']['X']['object']), passed=True)
            try:
                for key, description in entry['arrays'].items():
                    if description['object'] not in archive.verified:
                        a = archive.get(path, key)
                        if a.dtype.kind in 'fciub' and not np.isfinite(a).all():
                            raise ValueError(f'Nonfinite array {key}')
                        del a
                n, d = entry['arrays']['X']['shape']
                if n < 1 or d < 1:
                    raise ValueError('Empty feature matrix')
                if 'y' in entry['arrays'] and entry['arrays']['y']['shape'] not in ([n], [0]):
                    raise ValueError('Misaligned labels')
                if 'pairs' in entry['arrays']:
                    p = archive.get(path, 'pairs')
                    if p.ndim != 2 or p.shape[1] != 2 or p.dtype.kind not in 'iu' or np.any(p < 0) or np.any(p >= n) or np.any(p[:,0] >= p[:,1]):
                        raise ValueError('Invalid pair indices')
                    del p
            except Exception as exc:
                item.update(passed=False, error=f'{type(exc).__name__}: {exc}')
            result['other_dataset_containers'].append(item)
        precision_factors = {}
        for kind in ('geometry', 'precision'):
            count, failures = 0, []
            for path in (RECENT/'jobs').glob(f'{kind}_*/rows/*.json'):
                row = json.loads(path.read_text())
                expected = result['datasets'][row['dataset']]['manifest']['dataset']['sha256']
                if kind == 'geometry':
                    valid = row.get('dataset_sha256') == expected
                else:
                    # The precision worker records prefix size and quantization,
                    # rather than a raw dataset hash. Verify these against the
                    # independently validated and persisted input arrays above.
                    name = row['dataset']
                    key = (name, row['k'])
                    if key not in precision_factors:
                        center_file = RECENT/'jobs'/f'geometry_{name}'/f"centers_k{row['k']}_start3.npz"
                        with np.load(center_file, allow_pickle=False) as saved:
                            c = saved['centroids']
                            maximum = max(abs(result['datasets'][name]['statistics']['minimum']),
                                          abs(result['datasets'][name]['statistics']['maximum']))
                            precision_factors[key] = 4096 / max(maximum, float(np.max(np.abs(c))), 1e-15)
                    factor = precision_factors[key]
                    valid = row['n'] == min(128, result['datasets'][name]['statistics']['n']) and row['d'] == result['datasets'][name]['statistics']['d'] and row['start'] == 3 and row['input_factor'] == factor
                if not valid:
                    failures.append(str(path.relative_to(CODE)))
                count += 1
            result[f'{kind}_row_validation'] = dict(rows=count, mismatches=failures, passed=count > 0 and not failures,
                                                   check='dataset hash' if kind == 'geometry' else 'prefix size, original d, saved start and reconstructed quantization factor')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    for row_index, name in enumerate(('mnist', 'fashion_mnist')):
        if not (args.out/f'{name}_preview.npz').exists():
            continue
        with np.load(args.out/f'{name}_preview.npz', allow_pickle=False) as data:
            for label, ax in enumerate(axes[row_index]):
                ax.imshow(data['images'][label], cmap='gray', vmin=0, vmax=255)
                ax.set_title(f'class {label}', fontsize=9)
                ax.axis('off')
        axes[row_index,0].text(-.2, .5, name, transform=axes[row_index,0].transAxes, rotation=90, va='center')
    fig.suptitle('Preserved research subsets: first selected image of each true class')
    fig.tight_layout()
    fig.savefig(args.out/'image_sanity.png', dpi=150)
    plt.close(fig)
    checks = [c for r in result['datasets'].values() for c in r.get('checks', [])]
    result.update(complete=True, passed_checks=sum(c['passed'] for c in checks), failed_checks=sum(not c['passed'] for c in checks),
                  verified_archive_objects=len(archive.verified), wall_seconds=time.perf_counter()-start_time)
    result['all_passed'] = all(r['status'] == 'passed' for r in result['datasets'].values()) and all(r['passed'] for r in result['other_dataset_containers']) and result['geometry_row_validation']['passed'] and result['precision_row_validation']['passed']
    write_report(args.out, result)
    print(json.dumps({k: result[k] for k in ('complete', 'all_passed', 'passed_checks', 'failed_checks', 'verified_archive_objects', 'wall_seconds')}), flush=True)
    if not result['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
