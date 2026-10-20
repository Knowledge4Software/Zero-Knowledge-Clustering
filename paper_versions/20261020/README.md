# Three editable paper sources - 20 October 2026

Compile `manuscript.tex` inside the desired folder with `latexmk -pdf` using TeX Live:

- `01_supervisor_original/` preserves the historical first-draft manuscript bytes, original colors and placeholders. Its Springer class and figure dependencies are included. The historical bibliography is supplied from the project's pre-reuse revision.
- `02_changes_red_green/` contains the latest text with revised passages in red, added material in green and unchanged material in black. New bibliography entries are green. This is an editorial paragraph/environment comparison, not a word-level deletion redline.
- `03_final_revised/` is the clean AI-assisted paper in Elsevier elsarticle format, including the completed D/F results and numerical exports.

`change_map.csv` gives each comparison block's category and nearest original text. `ORIGINAL_PASSAGES_REWRITTEN_OR_REMOVED.txt` retains original passages not copied verbatim; some were rewritten rather than deleted. The original manuscript remains the authoritative baseline.

`validation.json` and `artifact_checks.json` record source identity, PDF hashes, build diagnostics and the completed checks. The evidence check files record captured phase counts and hash validation. `earlier_native_source_provenance.json` retains the earlier broad-native reconstruction record.

The final needs human scientific approval, completed author declarations, public artifact arrangements and independent saved reuse-proof replay before submission. Numerical exports and their input hashes are included; large raw experiment folders, datasets and proof bundles are separate. The journal-readiness assessment is provided alongside the source ZIP.
