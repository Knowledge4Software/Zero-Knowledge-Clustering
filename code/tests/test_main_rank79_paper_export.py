"""Publication gates: keep slow/fallback outcomes and refuse pending proofs."""
import json,shutil,tempfile
from pathlib import Path
import pytest
from analysis import export_main_rank79_paper as paper
from campaigns import main_rank79 as campaign
from commitjl.pilot_io import save


@pytest.fixture
def workspace_fixture():
    owner=(paper.REPO/'tmp').resolve();path=Path(tempfile.mkdtemp(prefix='paper_export_test_',dir=owner)).resolve()
    assert path.is_relative_to(owner)
    try:yield path
    finally:
        assert path.is_relative_to(owner) and path!=owner
        shutil.rmtree(path)


def test_export_retains_all_eleven_profiles_including_a_slower_fallback(workspace_fixture,monkeypatch):
    root=workspace_fixture;rep=root/'rep';scale=root/'scale';real_verify=campaign.verify_group
    old=json.loads((paper.REPO/'code/validation/outputs/group_compact_auxiliary_r79_n128_20261025/control/checked_comparison.json').read_text())
    rows=[];scaled=[]
    def add(target):
        r=dict(old);r.update(proposal_updates=4,conclusive_without_fallback=3,fallback_updates=1,full_reference_active_seconds=40000.,active_phase_seconds_including_fitting_and_fallback=60000.,full_over_group_active_phase=2/3,fallback_active_seconds=28000.,fresh_fallback_proofs=256)
        save(target/'checked_comparison.json',r)
        for n in ['public/public_manifest.json','public/group_verify.json','native_checks.json']:save(target/n,dict(test_only=True))
        return r
    for fit in ['fashion_case','mnist_case']:
        for variant in ['head79','head63','pca79']:
            r=add(rep/fit/variant);rows.append(dict(fit=fit,variant=variant,**r))
    for case in campaign.CASES:
        for variant in case['profiles']:
            r=add(scale/case['id']/variant);scaled.append(dict(case=case['id'],n=case['n'],variant=variant,**r))
    save(rep/'summary.json',dict(complete=True,results=rows));save(scale/'summary.json',dict(complete=True,results=scaled))
    def checked(path):
        path=Path(path)
        return json.loads((path/'checked_comparison.json').read_text()) if path.is_relative_to(root) else real_verify(path)
    monkeypatch.setattr(campaign,'verify_group',checked)
    monkeypatch.setattr(paper,'quality_for_group',lambda _:dict(mean_agreement=99.5,minimum_agreement=99.,mean_assignment_certified=90.,mean_change_width=1.,distinct_regressions=1,regressions_certified_without_fallback=0))
    result=paper.export(root/'generated',rep,scale,preview=False)
    assert result['main_proof_rows']==11 and not result['preview']
    table=(root/'generated/main_proofs.tex').read_text();metrics=(root/'generated/metrics.tex').read_text()
    assert table.count('0.67')==11 and 'Values below one denote a slower composed method' in table
    assert r'\def\MainRunsComplete{1}' in metrics and r'\providecommand{\MainHeadRatioMin}{0.67}' in metrics


def test_incomplete_main_campaign_cannot_enable_the_final_paper(workspace_fixture):
    root=workspace_fixture;save(root/'rep/summary.json',dict(complete=False));save(root/'scale/summary.json',dict(complete=True))
    with pytest.raises(AssertionError):paper.export(root/'generated',root/'rep',root/'scale',preview=False)
    assert not (root/'generated/metrics.tex').exists()
