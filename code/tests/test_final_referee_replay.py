"""The additional audit must replay, preserve receipts, and enable accounting."""
import pytest
from campaigns.final_referee_replay import replay_environment,run


def test_verification_environment_retains_frozen_moments_accounting_mode():
    env=replay_environment()
    assert env['COMMITJL_RESUME_PROOFS']=='1'
    assert env['GOMAXPROCS']=='1' and env['GOMEMLIMIT']=='256MiB'


@pytest.mark.parametrize('workers,memory',[(0,256),(9,256),(1,127),(1,1201)])
def test_invalid_replay_limits_rejected(workers,memory):
    with pytest.raises(ValueError):replay_environment(workers,memory)


def test_previous_audit_receipt_is_never_overwritten(tmp_path):
    audit=tmp_path/'audit';audit.mkdir();receipt=audit/'status.json';receipt.write_text('{"complete":false}')
    with pytest.raises(ValueError,match='fresh audit directory'):run(tmp_path,False,audit)
    assert receipt.read_text()=='{"complete":false}'
