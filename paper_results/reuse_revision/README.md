# Reusable-statistics manuscript

The canonical [manuscript](../../manuscript.tex) uses these authored introduction, method, security, experiment, discussion and appendix sections. `generated_20261020/` preserves the historical D/F analysis and its exact provenance; those timings remain explicitly historical and are not pooled with the completed reviewer-control cohort.

The [completed controls](../review_controls/README.md) supply the fair exact public-center baseline, actual 1,024-row proofs, integer accuracy certificates, matched PCA controls and higher-rank pilots. Their cost ratios compare exact and approximate fidelity contracts.

Build the current paper from the repository root:

```powershell
python code/analysis/build_reuse_manuscript.py --tag LOCAL_REVIEW_BUILD
```

Choose a fresh tag. The builder assembles portable source dependencies, compiles with TeX Live and reports LaTeX diagnostics. Rendered-page review is required before delivery. The earlier three-paper package is frozen separately under `paper_versions/20261020/`; routine draft building must not overwrite it.
