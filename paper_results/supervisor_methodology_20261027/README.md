# Methodology draft for supervisor review

This package contains a new methodology for the implemented, benchmarked
prototype-update certificate. It preserves both received ZIP archives and
does not modify the supervisor's abstract, introduction or author details.
The previously reviewed 15-page paper and its ongoing experiment queue are
also unchanged.

## Files

- `methodology.tex`: the replacement Methodology section.
- `methodology_pipeline.tex`: its original vector pipeline figure.
- `methodology_appendix.tex`: algebraic derivations, conditional fingerprint
  argument, integer guards and persisted basis recipe.
- `methodology_preview.tex`: a standalone Elsevier two-column wrapper.
- `methodology_references.bib`: bibliography for the excerpt.
- `prototype_update_methodology_20261027.pdf`: compiled excerpt in the ZIP.
- `methodology_checks.json`: build, preservation and numerical checks.
- `received_archives/`: the two original input ZIPs, unchanged, in the handoff ZIP.

In the workspace, verbatim extractions are under `received/` and
`style_reference/`; `received_sources_manifest.json` pins every extracted file.

## What the supplied files contain

`Sparse_JL_for_ZK_Clustering.zip` contains both a named-author, two-column
Elsevier `manuscript.tex` and a review-layout `manuscript_elsevier.tex` with
placeholder authors. Their abstract, introduction and related-work bodies
are identical. The ZIP's README describes a Springer manuscript, but that
description is stale: the main manuscript files use `elsarticle`; the separate
`sn-article.tex` is the Springer example.

The supplied title, abstract and introduction describe the earlier
commit-then-sample sparse-JL proposal. They also contain planned experiments,
unfilled quantitative results and geometric extensions such as DBSCAN.
That is scientifically different from the current implemented method.

The current method uses a training-fitted centroid-informed/PCA basis,
residual energy, authenticated exact old groups and moments, rounded-basis
switching intervals, bounded exact-row refinement, and paid exact fallback.
Its selected inputs have `tail_width=0`; the group circuit rejects a nonzero
tail. Randomized SVD used to fit a basis is not an oblivious JL embedding or
a JL accuracy guarantee. Historical filenames containing `jl` do not change
the implemented statement.

**Do not merge this section with the old JL title/abstract/introduction and
call the resulting paper consistent.** The supervisor should align those
sections and replace the old JL experimental plan and appendices. A title
consistent with the evidence is “Reusable Zero-Knowledge Certificates for
Prototype Updates.” No title or supervisor-authored section was changed here.

## Writing form followed

The Spectral CCD source was read, particularly its Methodology section:

1. A short methodological rationale and a formal problem statement.
2. A named method overview followed by numbered stages.
3. Inputs and symbols introduced before the equations using them.
4. An explanation of each equation's role and the data passed to the next stage.
5. Three numbered heading levels, with short inline paragraph labels.

The earlier writing rules are also followed: clear factual claims, limits
collected at the relevant guarantee boundary, established identities cited,
formal derivations in an appendix, and no invented experimental outcomes.
The prose and figure are original; no Spectral CCD scientific content is
copied. The methodology does not depend on unfinished runtime results.

The four stages are compact representation; authenticated groups/moments;
integer-safe switching bounds; and exact-row refinement/fallback. Prediction
agreement, mathematical label-certificate coverage and verified aggregate
update decisions are explicitly distinguished.

## Compiling and integrating

Compile this independent excerpt from its directory:

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error methodology_preview.tex
```

For the final full paper, replace the old Methodology body with
`\input{methodology.tex}` and add `\input{methodology_appendix.tex}` after
`\appendix`. Use the matching bibliography entries. The supplied main
manuscript already loads `amsmath`, `amssymb`, `amsthm` and defines
`proposition`. Add `\usepackage{tikz}` and
`\usetikzlibrary{positioning,arrows.meta}` for the pipeline. The excerpt's
wrapper also loads `hyperref` for clickable references. Copy all methodology
files beside the full paper's main TeX file, or adapt the input paths.

The older JL-specific appendices and experiments need replacement rather
than simple appending; their theorem references belong to the old protocol.
The handoff intentionally supplies a clean standalone section instead of
silently inserting incompatible mathematics into supervisor-authored prose.

## Source traceability

The method is checked against:

- `code/commitjl/reuse_certificates.py`: exact integer heads and energies.
- `code/commitjl/enriched_jl.py`: centroid-informed basis fitting.
- `code/experiments/rank79_independent_replication.py`: persisted 63/79 recipe.
- `code/commitjl/paired_certificates.py`: exact rounded-Gram coefficients.
- `code/commitjl/group_change_bounds.py`: exact old-group decomposition.
- `code/commitjl/relative_certificates.py`: native individual-label certificates.
- `code/zk/circuit/bound_group_compact.go`: linked moments and group query.
- `code/zk/circuit/bound_group_fingerprint.go`: row/fingerprint authentication.
- `code/zk/circuit/bound_group_refinement.go`: canonical exact corrections.

The public assigned aggregate can reveal group moments through authorized
probes. This disclosure is stated explicitly; the section does not claim
adaptive record privacy or that a proof of cost improvement proves better
classification accuracy.
