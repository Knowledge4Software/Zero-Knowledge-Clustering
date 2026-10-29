# Current evidence and verification

Current native records, immutable model/input pins, public proofs, verification
keys and checked timing receipts are retained in `outputs/`. Cleanup removes
inactive proving keys, serialized constraint systems and preparation caches;
these are rebuildable and are unnecessary for public proof replay. All caches
in the active two-review campaign and the queued resumable online-affine
control are preserved.

Important roots:

- `outputs/referee_two_reviews_20261027/`: current sensor/neural-text study,
  exact backend controls, learning-workload proofs and serial queue.
- `outputs/referee_optimized_20261026/`: retained frozen-profile image data,
  matched timing repetitions and the queued 1024-row affine continuation.
- `outputs/referee_final_20261025/`: earlier checked controls still used in
  the current paper; public artifacts and measured metadata remain.
- Other roots contain supporting refinements, negative tests and historical
  evidence cited by current source receipts. Their age does not make their
  public proofs or result records disposable.

From the repository root, with `PYTHONPATH` set to `code`:

```text
python -m analysis.two_review_campaign_status
python -m analysis.export_two_review_evidence
python -m pytest code/tests/test_two_review_evidence.py code/tests/test_referee_learning_workflows.py code/tests/test_optimized_evidence_export.py code/tests/test_online_affine_control.py -q
```

The current reviewed PDF validations are in `output/`. Regenerating exports
can incorporate newly completed jobs; it does not visually approve a new
PDF. Older fixed-date paper builders and layout checkers remain legacy
source, not current release commands.

The canonical dataset cache in `code/.cache/commitjl_data` remains available.
No connection to the retired Mobin server is permitted. Cleanup is actual
filesystem deletion, recorded in `output/project_cleanup_20261027.json`.
