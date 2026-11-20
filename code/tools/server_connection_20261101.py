"""Password-in-memory SSH access to the explicitly authorized November server.

The host key is pinned to the preceding inspection. No credentials are saved.
"""
import base64
import hashlib
import os
from pathlib import Path
import sys

DEPENDENCIES = Path(os.environ.get('TEMP', '/tmp')) / 'codex-ssh-check-20261101'
if DEPENDENCIES.exists():
    sys.path.insert(0, str(DEPENDENCIES))
import paramiko

HOST = '213.233.184.220'
HOST_KEY = 'XkGTFAoWzKmvV67efv4fRCslA3HUeNSXUjhULVfGg9k'
REMOTE = '/home/user01/zk-audit-20261101'


class PinnedHost(paramiko.MissingHostKeyPolicy):
    def missing_host_key(self, client, hostname, key):
        actual = base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')
        if hostname != HOST or actual != HOST_KEY:
            raise RuntimeError('Server host key does not match the inspection pin')


def connect(password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(PinnedHost())
    client.connect(HOST, port=22, username='user01', password=password,
                   look_for_keys=False, allow_agent=False, timeout=20,
                   auth_timeout=20, banner_timeout=20)
    return client


def execute(client, command, payload=None, timeout=45):
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    if payload is not None:
        stdin.write(payload)
    stdin.channel.shutdown_write()
    output = stdout.read().decode(errors='replace')
    error = stderr.read().decode(errors='replace')
    code = stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError(f'Remote command failed ({code}): {output}\n{error}')
    return output
