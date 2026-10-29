# Current manuscript sources and evidence

- [ccd_pilot_20261029/](ccd_pilot_20261029/README.md): active integrated
  methodology, CCD-structured preliminary evaluation and editable SVG figures.
- [referee_final_20261025/](referee_final_20261025/): preserved earlier sources;
  the root still includes its abstract, introduction and retained evidence.
- [supervisor_methodology_20261027/](supervisor_methodology_20261027/README.md):
  new staged methodology, derivations and unchanged supervisor/style inputs.
- [review_controls/](review_controls/README.md): retained numerical and proof
  evidence supporting current results, including negative and exploratory cases.

The [delivery index](../output/README.md) identifies current reviewed PDFs.
Obsolete manuscript directories, early table fragments and copied version
folders were deleted at the user's request. Current result records and proof
verification artifacts are retained under `code/validation/outputs/`.

Compile the full paper from the repository root with
`latexmk -pdf -outdir=tmp/pdfs/current manuscript.tex`. The methodology
excerpt has its own wrapper and integration instructions. A successful
compile does not constitute a new visual review or scientific validation.
