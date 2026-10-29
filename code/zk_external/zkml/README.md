# Published ZKML primitive adapter

Upstream: Chen et al., *ZKML: An Optimizing System for ML Inference in
Zero-Knowledge Proofs*, EuroSys 2024, DOI 10.1145/3627703.3650088.
Official source: https://github.com/uiuc-kang-lab/zkml .
Pinned upstream commit: `50d0cf30e2bbcb8b1f31839b775c1d018b720527`.

`prototype_audit.rs` adapts the upstream **exact dot-product, adder and
Poseidon chips and Halo2/KZG prover**, without using our compact group
certificate. It is not an unmodified published benchmark or a claim about
the complete ZKML compiler's optimal layout. The original compiler normalizes
arithmetic and uses a lookup-sized max gadget; applying that directly to the
current 12-bit original-space squared distances would either overflow its
lookup domain or change the audited integer objective. The adapter therefore
adds explicit 16-bit coordinate checks and 48-bit distance comparisons.

The catalog is committed in full with the upstream Poseidon permutation,
using injective 16-bit packing and a salted, dimension-bound hash chain.
This is an alternative owner-approved commitment to the same integers, not
the original gnark Poseidon2 Merkle root. All centers and the aggregate exact
cost are public. The circuit has the same shape and key for every subsequent
center bank. Every catalog record contributes exactly once. Original rows
and the salt remain private. No classification metric is inferred from
proof verification.

Setup and query are separate. Query runs reuse saved keys and parameters;
fresh standalone verification reads only the public bundle and independently
approved pin/task files. KZG parameters generated locally are for benchmark
execution only, not a production trusted-setup ceremony. Tampering checks
must pass before any timing comparison is accepted.

All matched method runs must use the same Linux/WSL environment, CPU worker
budget, catalog and candidate banks. Historical Windows times are not a
matched denominator. This adapter does not change the paper.

## Local reproduction

The isolated environment is under
`/home/koush/.local/share/commitjl-external/`, with Rust
`nightly-2025-03-01` and a separate Python virtual environment. It does not
replace the user's default Rust or Python setup. The checked-out upstream
source is `tmp/external_baselines_20261027/zkml/`; the new binary source is
copied to its `src/bin/prototype_audit.rs`. Add `sha2 = "0.10"` to that
checkout's dependencies for pin verification. The resolved Cargo lock is
retained in the experiment evidence.

Build with two compilation jobs, separately from timing:

```sh
export RUSTUP_HOME=/home/koush/.local/share/commitjl-external/rustup
export CARGO_HOME=/home/koush/.local/share/commitjl-external/cargo
export CARGO_TARGET_DIR=/home/koush/.cache/commitjl-external/zkml-target
export PATH="$CARGO_HOME/bin:$PATH"
cd tmp/external_baselines_20261027/zkml
CARGO_BUILD_JOBS=2 cargo +nightly-2025-03-01 build --release --bin prototype_audit
```

The Python environment needs numpy, psutil, pandas, scikit-learn,
matplotlib, msgpack and pycryptodome. Build the original Go command for
Linux from `code/zk/` and place it at
`tmp/external_baselines_20261027/commitjl-bound-linux`.

From the repository root, in the isolated Linux environment:

```sh
export PYTHONPATH="$PWD/code"
python -m campaigns.published_baselines_laptop
python -m analysis.published_baseline_report
```

The campaign is serial and resumable from checked phase receipts. A `STOP`
file in `code/validation/outputs/published_baselines_20261027/` requests a
stop. Failed attempts are retained. It runs real-proof and tampering smoke
checks first, then fixed 32-record Fashion and neural-text catalogs with
every supplied bank. These previously inspected fits are diagnostic
comparisons, not independent confirmation. The notebook/native numerical
metrics are never reported as proof timings.

`matched_plan.json`, `upstream_receipt.json`, `matched_queue.json`,
`resources/`, private inputs and public verification bundles remain under
that evidence directory. Do not refresh manuscript tables automatically.
