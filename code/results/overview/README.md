# Cross-RQ overview

`analysis/` holds the [plain-English author overview](analysis/project_overview_for_authors.md), [full numerical analysis](analysis/analysis.md), [paper-readiness review](analysis/paper_readiness_review.md), and run inventory. The four comparison PNGs live in their corresponding RQ analysis folders and appear in the author overview. `runs/` holds metadata or shared logs from the coordinated campaigns.

Regenerate the summaries from the per-RQ raw files with `python -m analysis.paper_results` and then `python -m analysis.author_overview` from `code/`. The scripts leave raw result rows unchanged.
