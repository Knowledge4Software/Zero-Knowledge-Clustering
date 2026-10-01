# Saved-result analysis

From `code/`, run:

```powershell
python -m analysis.paper_results
python -m analysis.author_overview
```

`paper_results.py` generates per-RQ statistics and figures plus the combined
evidence report. `author_overview.py` generates the four comparison charts and
the plain-English author overview. Outputs go to `results/rq*/analysis/` and
`results/overview/analysis/`. These commands do not rerun benchmarks or alter
the raw observations.

See the [results index](../results/README.md) for coverage and limitations.
