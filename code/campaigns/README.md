# Campaign implementations

The final server campaigns completed before rental expiry. No experiment
is running or automatically queued, and retired hosts must not be contacted.

Current-method source is retained for reproducibility:

- server_final_method_20261102.py: matched final-method/baseline tasks.
- paper_completion_20261102.py: component and operating-limit controls.
- paper_gap_checks_20261103.py: fixed-task paired timing measurements.
- review_completion_20261103.py: measured moments-first/Full service.
- review_fit_replications_20261103.py: additional numerical fits.
- extended_revision_20261103.py: larger proof cells and release checkpoints.
- proof_runtime.py: resource limits and execution accounting.

These encode Linux/WSL scheduling and historical frozen plans. They are not
a ready-made laptop queue. A new campaign needs isolated outputs, frozen
inputs, matched local baselines and appropriate memory/disk guards.

For the submitted results, see [Submitted evidence](../../output/README.md).
The packaged checks and public replay run locally.
