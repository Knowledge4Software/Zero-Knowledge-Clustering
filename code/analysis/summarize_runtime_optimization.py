"""Summarize short runtime tests without changing scientific paper timings."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import statistics


REPO = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def summarize(out):
    base = REPO / 'code/validation/outputs'
    families = []
    artifacts = []
    for family in ['row', 'switch', 'linked', 'full']:
        root = base / f'runtime_optimization_20261025_{family}'
        result = json.loads((root / 'result.json').read_text())
        samples = result['samples']
        assert len(samples) == 12 and all(s['verified'] for s in samples)
        assert len(list(root.glob('*.proof'))) == 12
        medians = {str(n): statistics.median(s['wall_seconds'] for s in samples if s['workers'] == n)
                   for n in [2, 4, 8]}
        families.append(dict(family=family, constraints=result['constraints'],
                             median_fresh_proof_and_replay_seconds=medians,
                             two_over_eight_workers=medians['2'] / medians['8'],
                             primary_timing=family != 'linked',
                             caveat=('Go tests/build overlapped the second pass; retain as exploratory only.'
                                     if family == 'linked' else 'Serial short microbenchmark; setup/load excluded.')))
        artifacts.extend(root.glob('*'))
    equivalence_root = base / 'runtime_optimization_20261025_equivalence'
    equivalence = json.loads((equivalence_root / 'equivalence.json').read_text())
    assert len(equivalence) == 3 and all(r['byte_identical'] for r in equivalence)
    cache_root = base / 'runtime_optimization_20261025_buffered_key_cache'
    cache = json.loads((cache_root / 'key_cache.json').read_text())
    assert cache['canonical_key_identical'] and cache['fresh_proof_verified']
    assert cache['encoding']=='compressed-buffered' and cache['optimized_bytes']==cache['compressed_bytes']
    rejected_root = base / 'runtime_optimization_20261025_key_cache'
    rejected = json.loads((rejected_root / 'key_cache.json').read_text())
    assert rejected['canonical_key_identical'] and rejected['fresh_proof_verified']
    digest_log = base/'runtime_optimization_20261025_digest.log'
    digest_allocations={}
    for name,ns,allocated in re.findall(r'BenchmarkKeyDigest/(\w+)-\d+\s+\d+\s+(\d+) ns/op\s+(\d+) B/op',digest_log.read_text()):
        digest_allocations[name]=dict(nanoseconds_per_operation=int(ns),allocated_bytes_per_operation=int(allocated))
    assert set(digest_allocations)=={'ReadFile','Streaming'}
    binary = REPO / 'code/zk/bin/commitjl-bound-group-runtime-optimized-20261025.exe'
    smoke_root = base / 'runtime_optimization_20261025_smoke'
    smoke = json.loads((smoke_root / 'checks.json').read_text())
    assert smoke['complete'] and smoke['binary_sha256'] == digest(binary)
    artifacts += list(equivalence_root.glob('*')) + list(cache_root.glob('*.json')) + list(rejected_root.glob('*.json'))
    artifacts += list(smoke_root.glob('*.json')) + [binary,digest_log]
    out.mkdir(parents=True, exist_ok=False)
    result = dict(complete=True, scope='Implementation microbenchmarks, not new end-to-end paper results.',
                  primary_timing_families=['row', 'switch', 'full'],
                  fresh_runtime_proofs_verified=48, fresh_checked_cache_proofs_verified=2,
                  fresh_smoke_proofs_verified=13, families=families,
                  circuit_equivalence=equivalence, key_cache=cache,
                  digest_microbenchmark=digest_allocations,
                  rejected_uncompressed_cache=dict(**rejected, adopted=False, reason='Slower checked reads and nearly twice the disk size; production encoding kept compressed.'),
                  new_binary_sha256=digest(binary),
                  runtime=dict(legacy_workers=2, tested_fast_workers=8, go_soft_memory_limit='1200MiB',
                               process_rss_guard_bytes=int(3.5 * 2**30), free_ram_reserve_bytes=650 * 2**20),
                  scientific_outputs_changed=False, main_campaigns_resumed=False,
                  limitations=['One real Fashion-MNIST configuration and representative fixed witnesses.',
                               'Four fresh proofs per CPU count; reversed pass order.',
                               'Existing keys loaded; no new end-to-end/cold-setup speed ratio.',
                               'Full accelerates similarly; relative method/Full advantage is not multiplied by this runtime gain.',
                               'The uncompressed-cache alternative was slower and rejected; production file formats remain compressed.',
                               'Frozen campaigns preserve their two-worker configuration and binary pins.'])
    (out / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (out / 'manifest.json').write_text(json.dumps(dict(sha256={p.relative_to(REPO).as_posix(): digest(p)
                                                             for p in artifacts if p.is_file()}), indent=2), encoding='utf-8')
    lines = ['# Runtime optimization: short validation', '',
             'The main campaigns remain user-paused. These measurements are separate from the manuscript.', '',
             '| Proof family | 2 workers (s) | 8 workers (s) | Speed ratio |',
             '|---|---:|---:|---:|']
    for row in families:
        if row['primary_timing']:
            seconds = row['median_fresh_proof_and_replay_seconds']
            lines.append(f"| {row['family']} | {seconds['2']:.3f} | {seconds['8']:.3f} | {row['two_over_eight_workers']:.2f}x |")
    lines += ['', 'The linked-change timing is retained in the JSON with a background-load caveat and excluded from this primary table.', '',
              'Loop-invariant bound calculations were hoisted out of circuit loops. All three newly compiled R1CS files are byte-identical to the frozen real-workload files. Assignment computation, certificate precision, endpoints and verifier obligations are unchanged.', '',
              f"Checked compressed proving-key read: {cache['compressed_unbuffered_read_seconds']:.3f}s with direct file reads versus {cache['optimized_read_seconds']:.3f}s with buffered reads. Buffered write: {cache['optimized_write_seconds']:.3f}s. This is a single key-load measurement, not an end-to-end comparison. Compressed file hashes match and a fresh proof verified.", '',
              f"Rejected alternative: uncompressed keys took {rejected['uncompressed_read_seconds']:.3f}s to read versus {rejected['compressed_read_seconds']:.3f}s compressed, with nearly twice the disk size. That encoding change was removed from production.", '',
              'Keys/proofs retain their compressed encoding. Binary I/O uses a 256 KiB buffer, flushes before fsync, and keeps normal library decoding checks and SHA256 file pins. File hashing streams instead of allocating the entire large key file.', '',
              f"Hash-memory tradeoff on this 118.8 MB key: {digest_allocations['ReadFile']['allocated_bytes_per_operation']:,} bytes allocated per hash before versus {digest_allocations['Streaming']['allocated_bytes_per_operation']:,} after. Standalone hash time was slightly slower (189 ms versus 202 ms), so this is a memory improvement, not a hash-throughput claim.", '',
              'New workloads can pin proof_runtime_workers=8 in protocol.json; local_fast_tradeoff exposes --workers 8 during --prepare. Group, Full and fallback inherit the same pinned count. A Full reference with a different worker count is rejected before copying. Historical protocols default to two workers, and paused campaigns were not modified.', '',
              'No new whole-run speedup or manuscript headline is inferred from these microbenchmarks.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(complete=True, report=str(out / 'REPORT.md'), families=families), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=REPO / 'code/validation/outputs/runtime_optimization_20261025')
    summarize(parser.parse_args().out.resolve())
