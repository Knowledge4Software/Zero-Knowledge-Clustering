# Unattended balanced paper queue

`campaigns/paper_queue.py` schedules the prepared RQ1–RQ5 workloads sequentially
under a server systemd service. Closing SSH, shutting down the laptop, or
closing this chat does not terminate the service. It does not require a
long-running chat/tool session after deployment.

The queue keeps the paper design at ten projection seeds and five clustering
starts, all eight declared RQ1 dataset shapes, four native workers with four
numerical threads each, and serial proof tasks with sixteen Go threads.
The engineering pilot uses three measured proofs; final proof tasks use ten.
No additional dataset or feature subset is introduced.

Order: RQ1 → RQ1 export → native RQ3/RQ4/RQ5 → selected RQ2 inventory → RQ2
engineering pilot → final RQ3/RQ5 proofs and ablations. Dependent phases are
skipped with a recorded reason if RQ1 fails; independent proof phases continue.
Scientific task failures remain recorded and are not retried automatically.

Safety limits are failure ceilings, not runtime predictions:

| Scope | Bound |
|---|---|
| RQ1 dataset worker | Four hours; 8 GiB RSS |
| Individual proof arm | Two hours; 40 GiB Go RSS |
| RQ2–RQ5 task | Four hours; 48 GiB tree RSS, or 8 GiB for parallel native tasks |
| Queue phase tree | 49 GiB RSS; 10 GiB disk free required |
| Idle phase | Stop after twenty minutes without CPU or top-level log progress |
| Phase wall time | RQ1 8h, export 1h, native 8h, selected 4h, pilot 24h, final proofs 24h |
| systemd service | 50 GiB memory cap, 72-hour final time cap; no automatic restart loop |

The queue is finite. A CPU-consuming loop is stopped by task/phase time limits;
an idle stall is stopped by the idle watchdog. Exceeding a limit preserves
existing evidence and marks failure. These controls cannot guarantee that
every scientific configuration succeeds or that the campaign finishes in six
hours. Six hours is a check-in time, not an automatic stop time.

The selected RQ2 runner retains its declared 8,192-input-element guard. All
eight original RQ1 dataset shapes exceed it; they are recorded as exclusions.
The engineering pilot and small-shape proof campaigns still execute real
proofs. This queue cannot complete the full original-dataset RQ2 proof table.
See [scope and dataset sizes](run_audit_dataset_runtime.md).

Deployment for this server uses:

- Service: `commitjl-paper-20261006.service`.
- Project: `/root/commitjl-server-check-20261006`.
- Results: `results/paper_server_20261006` inside that project.
- Controller state: `queue_state.json`; immutable commands: `queue_plan.json`.
- Logs: `controller.log`, `rq1.log`, `rq1_export.log`, `native.log`,
  `selected.log`, `pilot.log`, `proof.log`, and each scientific worker's log.
- Phase artifacts are in `rq1/`, `native/`, `selected/`, `pilot/`, `proof/`.

Read-only checks, while the experiments continue:

```bash
systemctl status commitjl-paper-20261006.service --no-pager
cd /root/commitjl-server-check-20261006
.venv/bin/python campaigns/paper_queue.py --status --root results/paper_server_20261006
tail -n 20 results/paper_server_20261006/controller.log
```

These commands do not stop, restart, retry, or export the running campaigns.
Do not start a second writer or change the measured source/config during the
campaign. The service is enabled for boot; an interrupted invocation can
resume saved evidence, while previously recorded failures remain visible.
