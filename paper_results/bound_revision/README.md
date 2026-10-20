# Current manuscript evidence: 18 October 2026

The canonical manuscript is `../../manuscript.tex`, with the original section structure retained. Authored experiment and protected-method
text is in `../experiments.tex` and `../protected_method.tex`. This directory contains the generated tables, full
precision CSVs, vector figures, and the SHA-256 provenance manifest for its
**bound-audit-v1** results. No legacy MiMC timing is used in the revised paper.

From the repository root, regenerate evidence with:

```powershell
python code/analysis/revise_paper_results.py
python code/analysis/export_targeted_paper_tables.py
```

Inputs are the local `code/results/final_paper_review_1792295389/` bundle.
The generator does not contact a server or launch experiments. It checks key
coverage counts, preserves both Full references, separates map density, matches
geometry cells, and exports exact source decimals behind the typeset tables.

Compile and synchronize the Springer/Overleaf copy with:

```powershell
python code/analysis/build_revised_manuscript.py
```

This requires Python and a working TeX Live installation with `latexmk`,
`pdflatex`, and BibTeX on PATH. Output PDF and a minimal source ZIP are written
under `output/`. Author names, affiliations, funding, competing interests and
publication declarations require author confirmation; no metadata is invented.

The table columns use scalar counts, not byte compression ratios. The original
arrays and complete proof artifacts are not all in this compact downloaded
bundle. Public archive/DOI assignment and release of the approved public-only
artifact remain author tasks.
