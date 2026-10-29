"""Record the user's explicit local-run authorization without reviving Mobin."""
from pathlib import Path
import json,socket,time
from campaigns.local_fast_tradeoff import save,sha

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'code/validation/outputs/referee_optimized_20261026'
POLICY=REPO/'code/validation/outputs/referee_round2_20261026/execution_policy.json'


def main():
    if (ROOT/'plan.json').exists():raise RuntimeError('Do not replace an existing frozen plan')
    policy=json.loads(POLICY.read_text());save(ROOT/'previous_execution_policy.json',policy)
    policy.update(updated_unix=time.time(),heavy_runs_allowed=True,prover_generation_allowed=True,
        automatic_experiment_resume_allowed=False,local_campaign_authorized=True,local_allowed_hostname=socket.gethostname(),
        replacement_execution_target='laptop',reason='User explicitly requested optimization, then running the remaining experiments and updating the paper on 26 October 2026, after retiring Mobin. This supersedes the earlier local hold for this bounded serial campaign.',
        local_resource_limits=dict(serial=True,go_soft_memory_mib=1200,maximum_process_rss_gib=3.5,minimum_commit_headroom_mib=1024,os_low_memory_stop=True,workers=8),
        mobin_server_retired=True,mobin_connection_attempts_allowed=False,remote_heavy_runs_allowed=False,remote_prover_generation_allowed=False,remote_allowed_hostnames=[])
    save(POLICY,policy)
    source=REPO/'code/validation/outputs/referee_round3_20261026/extended_study_plan.json';study=json.loads(source.read_text())
    study.update(execution_target='Explicitly authorized local laptop, serial resource-guarded campaign',
        execution_order=['Small real-proof and tamper smoke','Frozen-profile native independent fits and real-row scaling','Three matched Full/centroid/PCA repetitions plus online affine and PLONK','Resource preflight and larger matched proof cases'],
        inherited_study_plan_sha256=sha(source),implementation_status='Local campaign prepared; no scientific profile, seed, query, rank or exact-row cap changed.')
    save(ROOT/'study_plan.json',study)
    sources={p.relative_to(REPO).as_posix():sha(p) for p in [REPO/'code/commitjl/reuse_certificates.py',REPO/'code/commitjl/paired_certificates.py',REPO/'code/commitjl/group_change_bounds.py',REPO/'code/zk/cmd/commitjl-bound/main.go',REPO/'code/zk/cmd/commitjl-bound/resume_cache.go']}
    save(ROOT/'plan.json',dict(version=1,prepared_unix=time.time(),execution_target='laptop',workers=8,phase_wall_limit_seconds=12*3600,
        binary='code/zk/bin/commitjl-bound-final-optimized-20261026.exe',plonk_binary='code/zk/bin/full-plonk-control-local-20261026.exe',
        native_study_sha256=sha(ROOT/'study_plan.json'),source_sha256=sources,
        scientific_profile=study['frozen_profile'],selection='All predeclared outcomes retained. No rank/cap tuning or seed exclusion.',
        extended_proof_cases=[dict(id=fit+'_n'+str(n),fit=fit,n=n) for fit,n in [
            ('mnist_k5_seed173_age8',128),('fashion_mnist_k5_seed173_age8',1024),
            ('mnist_k10_seed173_age1',1024),('mnist_k20_seed173_age1',1024),
            ('mnist_k5_seed173_age1',4096),('mnist_k5_seed173_age1',16384)]],
        scope='New timing controls exclude frozen-model fitting consistently and never mix historical cold-fitting totals. Native kernel speed ratios are not proving speed ratios.',
        remaining_after_small_controls=['Complete real proof scaling where resource preflight permits','Specialized exact matrix-product protocol remains unimplemented','Production challenge service and independent security audit remain separate work']))
    print(json.dumps(dict(prepared=True,root=str(ROOT),remote_enabled=False)))


if __name__=='__main__':main()
