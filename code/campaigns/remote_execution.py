"""Honor the laptop hold while permitting the user's requested Mobin work."""
import os,platform,socket


def require_mobin(policy,*,system=None,hostname=None,target=None):
    if policy.get('mobin_server_retired',False):
        raise RuntimeError('Mobin is retired by the user; do not connect or launch work on that host. Heavy laptop runs remain held.')
    system=platform.system() if system is None else system
    hostname=socket.gethostname() if hostname is None else hostname
    target=os.environ.get('COMMITJL_EXECUTION_TARGET') if target is None else target
    allowed={v.lower() for v in policy.get('remote_allowed_hostnames',[])}
    if (not policy.get('remote_heavy_runs_allowed') or system!='Linux'
            or hostname.lower() not in allowed or target!='mobin'):
        raise RuntimeError('Heavy runs remain held on the laptop; use the authorized Mobin Linux host and explicit remote target')
