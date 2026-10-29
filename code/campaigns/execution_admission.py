"""Explicit research admission; the retired Mobin target stays disabled."""
import os,platform,socket


def require_execution(policy,*,target=None,system=None,hostname=None):
    target=os.environ.get('COMMITJL_EXECUTION_TARGET') if target is None else target
    system=platform.system() if system is None else system
    hostname=socket.gethostname() if hostname is None else hostname
    if target=='laptop':
        if (system!='Windows' or not policy.get('heavy_runs_allowed')
                or not policy.get('prover_generation_allowed')
                or not policy.get('local_campaign_authorized')
                or hostname.lower()!=policy.get('local_allowed_hostname','').lower()):
            raise RuntimeError('Local experiments require explicit current laptop authorization')
        return
    from campaigns.remote_execution import require_mobin
    require_mobin(policy,system=system,hostname=hostname,target=target)
