# Reusable Zero-Knowledge Certificates for Prototype Updates

This project audits whether a proposed prototype bank improves, worsens, or
preserves quantized nearest-center distortion on a committed catalog. The
proof service reuses authenticated reference assignments and group moments,
then applies switching bounds, exact-row corrections, and exact fallback.

## Submitted manuscript

[manuscript.tex](manuscript.tex) is the LaTeX source corresponding to the
submitted 29-page PDF. The four PDF figures it includes are in
[figures/](figures/). The bibliography is embedded in the source.

Compile from the repository root with pdfLaTeX or latexmk:

    latexmk -pdf -halt-on-error manuscript.tex

## Reproduce the results

The submission uses two versioned evidence packages. Their names and checking
instructions are in [output/README.md](output/README.md). Keep the packages
beside that README when working locally. The primary archive check validates
42 archive hashes and recomputes 400 objective changes. Public proof replay
covers the supplied verifier examples.

The [historical public evidence](https://huggingface.co/datasets/Parsagh1383/zk-audit-results)
is pinned in the paper to revision b244a3549bccb27fdcdbc1c0320458c4eada3284.

## Implementation

- [code/commitjl/](code/commitjl/) contains numerical bounds and planning.
- [code/zk/](code/zk/) contains Go circuits and public verification.
- [code/experiments/](code/experiments/) and [code/campaigns/](code/campaigns/)
  contain experimental implementations.
- [code/tests/](code/tests/) contains automated regression tests.

From the repository root in PowerShell, install the Python dependencies and
run the tests:

    python -m pip install -r code/requirements.txt
    $env:PYTHONPATH = Join-Path $PWD 'code'
    python -m pytest code/tests -q
