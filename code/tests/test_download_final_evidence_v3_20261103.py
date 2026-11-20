import base64
import gzip
import json

import pytest
from tools.download_final_evidence_v3_20261103 import decode_snapshot,configure_transport


def test_compressed_snapshot_roundtrip_and_reject_incomplete_metadata():
    data={'now':123.,'metadata':{'summary.json':'{"complete":true}'},'downloads':[]}
    encoded=base64.b64encode(gzip.compress(json.dumps(data).encode())).decode()
    assert decode_snapshot(encoded)==data
    incomplete=base64.b64encode(gzip.compress(b'{}')).decode()
    with pytest.raises(ValueError):decode_snapshot(incomplete)
    with pytest.raises(Exception):decode_snapshot('broken response')


def test_channel_open_timeout_and_keepalive_are_configured_before_sftp():
    class Transport:
        channel_timeout=3600
        def set_keepalive(self,value):self.keepalive=value
    transport=Transport()
    class Client:
        def get_transport(self):return transport
    assert configure_transport(Client()) is transport
    assert transport.channel_timeout==45 and transport.keepalive==15
