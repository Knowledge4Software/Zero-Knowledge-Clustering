# Cost-aware proof selection

The existing compact relation remains unchanged. `commitjl/cost_aware.py`
selects a proof plan before any changing-request proof is generated.

For a fixed committed catalog and old reference, `Planner` caches exact old
groups, moments, rounded-basis heads and residual energies once. Each candidate
gets the same exact-integer aggregate interval as `exact_old_correction_bounds`.
If inconclusive, the prover computes native exact distances and simulates up
to16 corrections in descending interval-width order. It uses compact proofs
only when the simulated answer is conclusive and estimated proof work beats
Full by the configured margin. Otherwise it selects Full immediately.

Native exact values only plan the witness work. They do not replace proofs.
Every selected correction is proved and every output is publicly verified.
If a selected compact plan nevertheless verifies as inconclusive, verification
fails; it cannot silently publish the planner's prediction. Public path choices
and refinement indices are observable output leakage.

`Planner.batch(centers, extra_setup_seconds=...)` also checks whether the
estimated savings cover extra compact setup, batch key-loading costs and a
batch-level safety margin. Use zero extra setup for already authenticated
cached summaries; key-loading overhead still applies to a new prover process.
This is a cost estimate, not a measured crossover guarantee.
Invalidate the cache when catalog, quantization, basis or old reference changes.

`validation/verify_cost_aware.py` takes independently retained owner/key pins
and an approved complete candidate list. The public routing lists map local
branch ordinals to global request IDs. The verifier requires disjoint complete
coverage, identical raw-catalog roots and matching coordinate conventions,
reconstructs both branch tasks, and freshly invokes the existing Go verifiers.
Full compares freshly verified costs against the approved old reference cost;
compact/refinement uses only the conclusive verified interval. The public
export excludes prepared rows, private moment tables, proving keys and caches.

From the repository root, with `PYTHONPATH=code`:

```text
python -m unittest discover -s code/tests -p test_cost_aware.py -v
python -m validation.check_cost_aware_proofs --root NEW_SMOKE_DIRECTORY --binary LINUX_BINARY
python -m campaigns.cost_aware_pilot --initialize --root NEW_PILOT_DIRECTORY
python -m campaigns.cost_aware_pilot --root NEW_PILOT_DIRECTORY
python -m analysis.cost_aware_report --root NEW_PILOT_DIRECTORY
```

For a new approved task on an existing Q8 pilot case, initialize and run the
reusable control instead of a new paired baseline campaign:

```text
python -m campaigns.run_cost_aware_control --root NEW_DIRECTORY --source COMPLETED_Q8_CASE --task APPROVED_TASK_JSON --initialize
python -m campaigns.run_cost_aware_control --root NEW_DIRECTORY
```

The task contains `comparison_base` and all approved `reuse_centers`. Reuse
requires the same catalog/model/key pins. The command produces new query
proofs and verified decisions; it does not claim a new baseline measurement.
The completed first pilot's original source is retained under its
`frozen_source/` directory. The final text follow-up is reported by
`python -m analysis.cost_aware_final_report`.

The proof pilot runs in WSL with the existing binary and Python environment.
It uses the earlier Q8 pilot to freeze costs, then produces fresh warm query
proofs for text16 and Fashion8 against a fresh warm Full arm. Both arms reuse
immutable setup and applicable old evidence. Main generation is serial,8
workers,3.5GiB process-tree RSS limit. The recorded phase totals charge key
loading, proof/witness generation, native planning and public replay; they
exclude copying/export and independent oracle checks. Pair the run with the
Windows memory and scoped awake guards, using its newly written UTC deadline.
Never reuse an expired deadline or overwrite an earlier attempt receipt.

The planner is basis-agnostic and also applies to PCA. It adds no new
classification-accuracy or mathematical separation claim. The existing frozen
data is exploratory. New fits and repeated timings are still required for a
paper-level performance claim. The paper and previous evidence are preserved.
