import pytest
from campaigns.remote_execution import require_mobin

POLICY=dict(remote_heavy_runs_allowed=True,remote_allowed_hostnames=['mobin','Cloudfpsir'])


def test_authorized_remote_execution():
    require_mobin(POLICY,system='Linux',hostname='mobin',target='mobin')


@pytest.mark.parametrize('system,host,target',[
    ('Windows','mobin','mobin'),('Linux','laptop','mobin'),
    ('Linux','mobin',None),('Linux','laptop-wsl','mobin')])
def test_laptop_and_implicit_execution_remain_held(system,host,target):
    with pytest.raises(RuntimeError):require_mobin(POLICY,system=system,hostname=host,target=target)


def test_policy_can_hold_even_correct_host():
    with pytest.raises(RuntimeError):require_mobin(dict(POLICY,remote_heavy_runs_allowed=False),system='Linux',hostname='mobin',target='mobin')


def test_retired_server_is_rejected_even_with_stale_allow_flags():
    with pytest.raises(RuntimeError,match='retired'):
        require_mobin(dict(POLICY,mobin_server_retired=True),system='Linux',hostname='mobin',target='mobin')
