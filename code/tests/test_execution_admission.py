import pytest
from campaigns.execution_admission import require_execution


def test_retired_remote_does_not_block_new_explicit_local_authorization():
    policy=dict(mobin_server_retired=True,heavy_runs_allowed=True,
        prover_generation_allowed=True,local_campaign_authorized=True,local_allowed_hostname='study-pc')
    require_execution(policy,target='laptop',system='Windows',hostname='STUDY-PC')
    for updates in [dict(heavy_runs_allowed=False),dict(prover_generation_allowed=False),dict(local_campaign_authorized=False)]:
        with pytest.raises(RuntimeError):require_execution(dict(policy,**updates),target='laptop',system='Windows',hostname='study-pc')
    with pytest.raises(RuntimeError):require_execution(policy,target='mobin',system='Linux',hostname='mobin')
    with pytest.raises(RuntimeError):require_execution(policy,target='laptop',system='Windows',hostname='other-pc')
