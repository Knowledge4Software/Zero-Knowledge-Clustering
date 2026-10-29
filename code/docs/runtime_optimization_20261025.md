# Runtime optimization, 25 October 2026

Main research campaigns remain **user-paused**. Only short implementation
benchmarks and verification tests were authorized for this task. The manuscript
and its recorded deployment timings are unchanged.

Measured implementation changes:

- The real Fashion-MNIST row and switch proof circuits take approximately one
  third of the proving time with eight Go workers instead of two. Full also
  benefits similarly. These are representative-proof measurements using loaded
  frozen keys, not new whole-deployment speed ratios.
- Rounded-basis limits were recomputed inside the coordinate loops of the
  linked moment/change and switching circuits. Computing those constants once
  reduces compilation work. All three newly compiled R1CS files are byte-for-byte
  identical to the frozen real-workload files. No circuit constraints were removed.
- Binary key/constraint/proof reads and writes now use a 256 KiB buffer. Writes
  still flush before fsync; decoding checks and SHA256 pins remain in place.
  A checked 118.8 MB compressed key loaded in 16.16 s versus 30.84 s with direct
  file reads. This is one key-load measurement. Its buffered serialized bytes
  match the original file exactly, and a fresh proof verified.
- File SHA256 streams from disk instead of allocating the entire key file.

An uncompressed key-cache experiment was **rejected**: checked reads were slower
(39.17 s versus 31.38 s), and the file was nearly twice as large. Production
encoding remains compressed. Per-proof forced GC measured only milliseconds,
so the existing memory policy was retained.

See the generated [measurement report](../validation/outputs/runtime_optimization_20261025/REPORT.md)
and its JSON/hash manifest. Linked-change proof timings retain a background Go
test/build overlap warning and are excluded from the primary speed table.

The release executable is
`code/zk/bin/commitjl-bound-group-runtime-optimized-20261025.exe`. It is separately
built; previous frozen executables and all saved proof bundles are preserved.
Its full compact-protocol smoke includes 13 fresh publicly replayed proofs and
12 rejected tampering cases.

## Using the faster configuration

New workloads can pin `"proof_runtime_workers": 8` in their `protocol.json`.
`campaigns.local_fast_tradeoff --prepare --workers 8` exposes this setting.
Group, Full and prospective fallback inherit the same pinned count and keep
the 1200 MiB Go soft memory limit, 3.5 GiB process RSS guard and 650 MiB free-RAM
reserve. Full reference copying rejects different worker counts before copying
files. Historical protocols without this field retain two workers regardless
of host environment overrides.

Do not edit a paused protocol or replace its frozen binary. Any new accelerated
timing comparison needs a separate matched configuration; do not compare an
eight-worker method against historical two-worker Full as a new speed claim.
Completed scientific results remain usable without rerunning them. The main
coordinators and deferred paper updater were not restarted.

## Reproducing the short checks

`cmd/commitjl-bound/performance_test.go` has explicitly gated tests. Ordinary Go
test runs skip real-workload microbenchmarks. To run them, compile a Go test
binary and set `COMMITJL_BENCH_SOURCE` to a completed control directory and
`COMMITJL_BENCH_OUTPUT` to a new, separate output directory. Families are selected
with `COMMITJL_BENCH_FAMILY=row|switch|linked|full`; the test generates four fresh
proofs per CPU count across reversed passes. Use the normal 1200 MiB Go limit.

`TestFrozenCircuitEquivalence` recompiles the three compact circuits and checks
their serialized SHA256 against the frozen caches, with two workers.
`COMMITJL_BENCH_BUFFERED_KEY_CACHE=1` selects the checked compressed-key I/O test.
`COMMITJL_BENCH_KEY_CACHE=1` reproduces the rejected uncompressed alternative.
Neither test resumes a campaign or changes source receipts.
