"""Read-only method/evidence audit and structural validation of the rental plan."""
import json
from pathlib import Path

from analysis.update_latest_paper_20261029 import BUILD, GATE, PAPER, REPO, read, sha, verify_preservation, write


def main():
    pins = verify_preservation()
    gate = read(REPO / 'output/published_baseline_gate_20261029.json')
    assert gate['complete'] and gate['proof_campaign_complete'] and gate['warm_comparison_complete']
    binary = REPO / 'tmp/external_baselines_20261027/commitjl-bound-linux'
    binary_hash = sha(binary)
    for role in ['compact_smoke', 'refinement_smoke']:
        smoke = read(GATE / 'proofs' / role / 'checks.json')
        assert smoke['complete'] and smoke['public_only'] and smoke['binary_sha256'] == binary_hash
    profiles = []
    for dataset in ['fashion', 'text']:
        source = GATE / 'proofs/matched' / (dataset + '_n32') / 'ordered_head'
        model = read(source / 'head_r63/input.json')['model']
        aux = read(source / 'auxiliary_basis.json')
        assert len(model['basis']) == 63 and len(aux['certificate_basis']) == 79
        assert aux['certificate_basis'][:63] == model['basis']
        assert model['basis_scale'] == 16384 and model['input_bits'] == 12
        assert model['tail_width'] == 0 and model['tail_rows'] == [] and model['tail_signs'] == []
        profiles.append(dict(dataset=dataset, prediction_rank=63, certificate_rank=79,
            basis_scale=16384, input_bits=12, JL_tail=False, correction_cap=16))
    native = read(GATE / 'native_results.json')
    assert native['all_deterministic_intervals_valid'] and native['no_wrong_certified_labels']
    assert read(GATE / 'warm_followup_v2/checks.json')['complete']
    tests = read(BUILD / 'python_tests.resource.json')
    assert tests['complete'] and tests['returncode'] == 0
    test_summary = (BUILD / 'python_tests.log').read_text().strip().splitlines()[-1]
    go = read(BUILD / 'go_tests.json')
    assert go['returncode'] == 0 and go['packages_passed'] == ['commitjl/zk/circuit', 'commitjl/zk/cmd/commitjl-bound']
    plan = read(REPO / 'code/experiments/server_balanced_72h_20261029.json')
    phases = plan['phase_budgets']
    assert phases[0]['start_hour'] == 0
    assert all(a['end_hour'] == b['start_hour'] for a, b in zip(phases, phases[1:]))
    assert phases[-1]['end_hour'] == plan['initial_campaign_target_hours'] == 32
    assert 30 <= plan['initial_campaign_target_hours'] <= plan['initial_campaign_hard_cap_hours'] <= 40
    assert plan['rental_hours'] == 72 and not plan['extension_possible'] and plan['final_paid_allocation']
    assert plan['minimum_uncommitted_reserve_hours'] == plan['rental_hours'] - plan['initial_campaign_hard_cap_hours']
    assert plan['recovery_policy']['proof_job_start_cutoff_hour'] + plan['maximum_individual_job_minutes']/60 <= 64
    assert plan['new_proving_job_start_cutoff_hours'] + plan['maximum_individual_job_minutes']/60 <= plan['proof_job_drain_deadline_hours']
    assert plan['long_case']['checkpoints'] == sorted(set(plan['long_case']['checkpoints']))
    checks = dict(complete=True, total_hours=32, initial_campaign_hard_cap_hours=36,
        rental_hours=72, minimum_uncommitted_reserve_hours=36, extension_assumed=False,
        primary_target_runs=48, primary_n=128, primary_q=4, long_case_q=32,
        new_proofs_or_remote_jobs=0, validation_scope='Structural plan checks, not an executable campaign readiness certificate.',
        plan_status=plan['status'], ready_to_start_billing=plan['purchase_gate']['ready_to_start_billing'])
    write(REPO / 'output/server_balanced_72h_20261029_checks.json', checks)
    value = dict(complete=True, current_selected_method='Authenticated exact old groups/moments, integer-safe rounded-basis switching intervals, bounded verified row refinement and fresh proved Full fallback.',
        latest_selected_profile_consistent=True, profiles=profiles, core_and_protected_authoring_unchanged=True,
        protected_files=len(pins['protected_sha256']), proof_binary_sha256=binary_hash,
        proof_binary_matches_checked_smokes=True, native_integer_checks_passed=True,
        completed_cold_and_warm_public_replay_passed=True, python_test_summary=test_summary, go_checks=go,
        method_changed=False, scope='Current selected repository method. This is not a claim of newest/best published algorithm, production security audit or publication readiness.',
        evidence_assessment='Coherent and useful regime-specific verification mechanism; insufficient for universal superiority, a consistent PCA advantage, general text benefit or a confident journal-readiness claim.',
        remaining_evidence=['Repeated paired real-proof timings across independent fits',
            'Matched larger-catalog proof comparisons', 'Complete-response preparation/reuse/reference-reset accounting'],
        server_plan_checks=checks)
    write(REPO / 'output/current_method_audit_20261029.json', value)
    text = f'''# Audit of the current selected method

The active paper and completed proof gate use the latest selected four-stage
repository method: authenticated exact old groups and moments, an integer-safe
rounded-basis switching interval, up to16 verified exact-row corrections and
fresh proved Full fallback. Prediction rank63, certificate rank79, scale16384
and 12-bit inputs agree in both proof cohorts. No JL tail is present.

The methodology and appendix remain byte-identical to the supervisor sources.
The abstract, introduction, methodology figures, master manuscript and core
Python/Go files are unchanged across this refresh ({value['protected_files']}
protected files). Existing bibliography bytes are preserved; three references
are appended. The measured Linux binary matches the retained compact and
refinement public-only smoke receipts by SHA256. Its build is Go1.27.1,
gnark0.16.3 and gnark-crypto0.21.0; current-source circuit/command tests also pass.
Matching smoke binary hashes is not a claim of bit-identical recompilation.

Fresh numerical checks: **{test_summary}**. Current-source Go checks passed in
both circuit and command packages, covering exact old-group authentication,
commit-before-challenge, linked moments, switching, refinement, public task
coverage and a tiny Groth16 public replay. That replay is a validation test,
not an additional benchmark timing sample. All completed cold/warm comparison
verification receipts and exact-integer oracles pass. These checks do not
replace an external security audit or production challenge/setup governance.

The method is technically coherent enough to evaluate without changing its
relations. The evidence is still preliminary for strong comparative/journal
claims: PCA is close, the cold ZKML adapter wins, and text routes to Full.
Better label agreement does not establish better aggregate sign estimation.
The useful supported claim is selective verified reuse on the recorded image
regime, with explicit disclosure and fallback costs.

The one paid rental now targets32 initial hours, caps them at36, and leaves
at least36 hours for recovery/final validation and transfer. Server runtime
is unmeasured and the full executable campaign is not yet certified ready.
More RAM/cores cannot guarantee favorable scientific outcomes. No server or
new benchmark proof campaign was started for this audit.
'''
    write(REPO / 'output/current_method_audit_20261029.md', text)
    print(json.dumps(dict(complete=True, python_tests=test_summary, protected_files=value['protected_files'], initial_campaign_hours=32)))


if __name__ == '__main__':
    main()
