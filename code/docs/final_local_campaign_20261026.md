# Current local campaign, 26 October 2026

27October later observation: the recorded scheduler12684, affine worker22244
and its provers no longer exist. No error explaining the termination was
recorded, so the cause is unconfirmed. Root/cohort statuses are now marked
interrupted instead of leaving a stale running label. Original queue files
and the observation are preserved in `runtime_interruption_20261027/`.
Affine bank000 is completed and bank001 stage1 is partially generated;
its resource receipt and sample checkpoints must be retained and charged
if an explicitly requested continuation resumes this current campaign.
No old experiment has been automatically resumed.
The latest reviewed paper is `output/pdf/prototype_audit_optimized_20261027.pdf`,
with17 visually inspected pages and21 completed new method--Full pairs.

27 October scope amendment: the user declined a10-20day proof campaign.
Further proof catalogs are capped at1024 records. The4096/16384 proof
benchmarks are deferred before starting; completed native results at those
sizes remain available and do not establish proof speed. The original plan
is retained, alongside `laptop_scope_amendment_20261027.json` in the campaign
root, explicitly recording that this decision followed partial results.

The bounded scheduler `campaigns.bounded_local_campaign` replaced only the
two Python scheduling processes. It adopted the same active online-affine
worker and prover, preserving completed phases and their costs. Current
Fashion-MNIST N1024 work continues for three matched repetitions and online
affine. Other declared N<=1024 cases proceed only through a guarded,
five-minute compile preflight, then the existing per-command RAM/disk/time
limits. Oversized circuits and memory exclusions remain in the evidence.
No old queue is automatically restarted and no scientific profile changes.
Scheduler handover identity, original queues and policy are preserved in
`scope_handover_20261027/`. The following large-catalog runtime discussion
describes the superseded plan; it does not authorize a larger proof run.

The user explicitly authorized optimization, remaining experiments and paper
revision after retiring Mobin. Local serial execution now supersedes the old
laptop hold for this campaign. Mobin remains retired; never reconnect.

Selected scientific profile is frozen: prediction63, certificate79,
precision14, at most16 exact rows in canonical width order. PCA and the
centroid-informed basis use the same certificate. Optimizations preserve
integer results and proof relations; they do not improve prediction accuracy.

Completed:56 representation fits on8 dataset/seed splits,280 retained native
case records, k3/5/10/20 and real catalogs through16384. At k5,N1024 the
centroid basis resolves60/60 updates and5/5 natural regressions. Its mean
agreement is99.942% MNIST and99.948% Fashion. Fashion k10 leaves2/8 updates
inconclusive, including1/3 regressions. MNIST k20 includes one centroid
request below99% agreement. These failures remain in the paper and evidence.

The 16384-row primary age1 MNIST case resolves4/4, with99.786% mean and
99.762% worst agreement. Native checks are not proof performance.

Current queue:
`code/validation/outputs/referee_optimized_20261026/queue.json`.
The original live matched worker was adopted by
`campaigns.final_local_supervisor`; no proof process was restarted. This
corrects a whole-series12hour cap. Small commands keep12hour limits;
declared4096/16384-row commands allow72hours equally for all methods and
fallback, with unchanged resource guards. The large-catalog amendment
records why healthy serial proof generation can exceed a day.
Use `python -m analysis.optimized_campaign_status` to check actual processes.

Controls: three fresh matched Full/centroid/PCA/moments repetitions, followed
by declared independent-fit, k and real-row scale cases. Online affine sees
only the current center set and is rebuilt and paid each time. The PLONK
control uses an explicitly unsafe test SRS, as a backend timing control only.
Both have passed small complete proof/replay checks. Production challenge
service, stronger aggregate privacy and a specialized exact matrix-product
protocol remain outside the implemented fixes.

The additional Fashion k10/k20 proof stress cases were added after the native
study. Their amendment explicitly marks them diagnostic, not independent
confirmation. Original case membership, ranks and cap remain unchanged.

RAM: one experiment at a time,8 Go workers for matched methods,1200MiB Go
soft target,3.5GiB process RSS cap, Windows low-memory/1GiB commit reserve.
Disk guards retain2GiB before new phases and1GiB in flight. User cleanup
freed approximately88GiB. Resource exclusions and all failed attempts remain.
The original largest Full case alone required10240 batch proofs per
repetition. It is now deferred, so this campaign no longer includes that
week-plus workload.

Create a file named `STOP` in the campaign root to stop only this job tree.
Do not change protocol/input/binary pins in place or automatically restart
old paused campaigns. A completed comparison triggers evidence export and
an automatic PDF candidate. Candidates require visual review; the reviewed
private PDF remains separate. Nothing is pushed or made public.
