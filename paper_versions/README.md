# Paper versions

`20261020/` is the preserved earlier three-version delivery:

- `01_supervisor_original/`: preserved original manuscript source and its build dependencies.
- `02_changes_red_green/`: earlier revised text with editorial comparison colors.
- `03_final_revised/`: clean AI-assisted revision and numerical provenance.
- `change_map.csv`: paragraph/environment classifications and nearest original text for review.
- `ORIGINAL_PASSAGES_REWRITTEN_OR_REMOVED.txt`: original passages not retained verbatim; the authoritative original remains unchanged.
- `validation.json`: source identity, PDF hashes, build diagnostics and visual-review status.

The editable working manuscript is `../manuscript.tex`. The current completed-results revision is separate; see `../output/README.md`. Do not overwrite this frozen historical release during routine builds.

The comparison is deliberately at paragraph, equation and table level. Red denotes revised passages, green denotes added material, and black denotes unchanged material. It is not a word-level tracked-deletion rendering. Document-format changes from Springer to Elsevier are not treated as scientific changes.

The user confirmed on 20 October 2026 that the preserved `paper_history/supervisor_draft.tex` is the intended supervisor-original baseline. Its manuscript bytes are kept unchanged.
