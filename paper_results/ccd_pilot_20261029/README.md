# CCD-structured preliminary manuscript, 29 October 2026

## Latest baseline refresh

The current delivery is
`output/pdf/prototype_audit_latest_baselines_20261029.pdf` (20 pages), with
matching `_sources.zip` and `_validation.json` files in `output/`.
Five additional generated tables incorporate the completed three-fit native
gate, paper-derived random projections/coresets/RaBitQ, the fresh cold proof
cohort and the equal-routing warm follow-up. Native and proof measurements
remain separate. PCA ties, cold adapter wins and unfavorable text outcomes
are explicit; no server measurement or new benchmark proof is invented.

The protected abstract, introduction, methodology, appendix and method figures
remain unchanged. `latest_evidence_manifest.json` pins the added exports.
The current-method audit also checks the active profile and retained proof
binary, with 39 fresh Python tests and selected Go circuit/command checks.

For this refresh use `python code/analysis/update_latest_paper_20261029.py build`.
Its evidence/package commands require the full private workspace and the
already captured preservation baseline; do not recapture that baseline after
editing. Ordinary LaTeX compilation from the source ZIP needs neither audit
scripts nor private proving artifacts. The older integration/build notes below
describe the retained original 18-page pilot delivery.

The active master is [manuscript.tex](../../manuscript.tex). It integrates the
completed four-stage methodology with a new post-methodology draft following
the section order of the supplied Spectral CCD paper:

1. Experimental Design: four research questions, datasets, baselines, protocol,
   settings and metrics.
2. Experimental Results: component roles, preparation and request work,
   measured reuse, catalog size, proof material and difficult requests.
3. Discussion, Threats to Validity and Conclusions.
4. The unchanged methodology appendix and a supporting-evidence appendix.

The section pattern follows the supplied style reference; the new scientific
prose describes this project. No CCD results or scientific claims are imported.

## Preservation

The earlier master is retained in
`paper_history/ccd_pilot_before_20261029/manuscript.tex`, with a SHA256
preservation receipt. The original abstract, introduction and earlier master
prefix are unchanged. `methodology.tex` and `methodology_appendix.tex` here are
byte-identical copies of the completed supervisor-methodology sources.
The original earlier authoring files and both supplied source ZIPs are unchanged.
Only the integrated methodology figure implementation and additional figure
input are new; completed methodology prose and equations are preserved.

## Evidence status

This is a preliminary paper for review before the proposed server campaign.
No new experiments, proofs, remote actions or server measurements are performed
by this manuscript update. Six new tables are generated from eleven completed
local evidence exports, pinned in `evidence_manifest.json`.

The draft retains the weak text results, near-ties with default-rank PCA, the
incomplete PCA128 attempt, the mixed-route slowdown and the unpaired status of
the final text rerun. Native ablations are not described as proof timings.
Preparation allocation is distinguished from independently measured hot
response latency; measured warm results do not establish a cold crossover.
Larger server runs and longer request sequences remain prospective.

## Figures

`figures/methodology_four_stages.svg` and
`figures/methodology_interval_refinement.svg` are the two new methodology
figures. Matching vector PDFs are included by LaTeX; PNGs are previews.
Three more editable figures show measured warm pilots, native rank coverage
and the separate cost-aware scheduling variant. Interval endpoints in the
methodology illustration are schematic, not observations.

## Build and regeneration

From the repository root:

```powershell
python code/analysis/prepare_ccd_pilot_20261029.py evidence
python code/figures/ccd_pilot_methodology_20261029.py
python code/analysis/prepare_ccd_pilot_20261029.py build
```

The `prepare` phase records and integrates the earlier master once; do not
rerun it against this already integrated master. The ordinary PDF build is:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
```

The source ZIP contains the active LaTeX dependency closure, editable figures,
numerical exports used by the new tables, figure/table generators and build
instructions. It is a paper build package, not the complete private proof
archive or an independently executable proof campaign. Retained aggregate
reports refer to proof artifacts in the main private repository.

Earlier automated authoring/finalizer scripts target the older manuscript
generation and must not be run to overwrite this reviewed draft. Their older
authoring pins are not silently reissued by this paper-only update.

Author metadata and final submission declarations still require the authors'
review. A successful build does not make preliminary pilot findings final.
