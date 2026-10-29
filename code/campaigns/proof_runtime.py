"""Pinned proof parallelism, shared by methods and their Full/fallback baselines.

Historical protocols without this field keep two Go workers. Increasing it is
an explicit new experiment configuration, never a change to a paused campaign.
The existing 1200 MiB Go soft limit and external RAM/RSS guards are retained.
"""
import os
import psutil


def phase_limit(protocol,default=4*3600):
    """Keep historical caps; allow the explicitly declared real-row series.

    Tens of thousands of proofs can take more than a day without a stalled
    worker. RAM, disk and user-stop guards remain independent of this cap.
    """
    value=protocol.get('phase_wall_limit_seconds',default)
    large=(protocol.get('long_catalog_phase')=='local_20261026' and protocol.get('n',0)>=4096)
    maximum=(72 if large else 24)*3600
    if type(value) not in (int,float) or not 60<=value<=maximum:
        raise ValueError('phase wall limit outside the declared catalog runtime budget')
    return float(value)


def proof_workers(protocol):
    workers = protocol.get('proof_runtime_workers', 2)
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError('proof_runtime_workers must be an integer from 1 to 64')
    return workers


def proof_environment(protocol):
    return dict(os.environ, COMMITJL_RESUME_PROOFS='1',
                GOMAXPROCS=str(proof_workers(protocol)), GOMEMLIMIT='1200MiB',
                OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')


def check_reference_runtime(source, target):
    if proof_workers(source) != proof_workers(target):
        raise ValueError('Full reference and method use different proof worker counts; generate a matched baseline')


def launch_headroom(previous):
    """Scheduling only; the in-flight 650 MiB/RSS guards remain mandatory.

    A single explicit attempt can let Windows reclaim its file-cache working
    set normally. After a memory-guard stop revert to the conservative observed
    peak, avoiding a repeated launch/kill loop on a busy laptop.
    """
    if os.environ.get('COMMITJL_MEMORY_PRESSURE_POLICY') == 'windows_os':
        if os.name != 'nt':raise ValueError('windows_os memory policy requires Windows')
        # A low-memory kill frees the child's own allocation. That transient
        # availability must not immediately authorize another identical kill.
        if any(a.get('stop') == 'RAM reserve' for a in previous.get('attempts', [])):
            return max(1500*2**20, previous.get('peak_rss_bytes', 0)+768*2**20)
        return 768*2**20
    conservative = max(1500*2**20, previous.get('peak_rss_bytes', 0)+(650+256)*2**20)
    requested = os.environ.get('COMMITJL_LAUNCH_AVAILABLE_MIB')
    if requested is None:
        return conservative
    value = int(requested)
    if not 1152 <= value <= 16384:
        raise ValueError('Launch-only headroom must be at least 1152 MiB; in-flight guards cannot be relaxed')
    if any(a.get('stop') in ['RAM reserve', 'process RSS guard'] for a in previous.get('attempts', [])):
        return conservative
    return value*2**20


_memory_notification = None


def _windows_memory_pressure():
    """Read-only OS low-memory notification plus system commit headroom."""
    import ctypes
    from ctypes import wintypes
    import atexit
    global _memory_notification
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateMemoryResourceNotification.argtypes=[ctypes.c_int]
    kernel.CreateMemoryResourceNotification.restype=wintypes.HANDLE
    kernel.QueryMemoryResourceNotification.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL)]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    if _memory_notification is None:
        _memory_notification=kernel.CreateMemoryResourceNotification(0)
        if not _memory_notification:raise OSError(ctypes.get_last_error(),'Memory notification unavailable')
        atexit.register(kernel.CloseHandle,_memory_notification)
    low=wintypes.BOOL()
    if not kernel.QueryMemoryResourceNotification(_memory_notification,ctypes.byref(low)):
        raise OSError(ctypes.get_last_error(),'Memory pressure query failed')
    class MemoryStatus(ctypes.Structure):
        _fields_=[('length',wintypes.DWORD),('load',wintypes.DWORD)]+[(name,ctypes.c_ulonglong) for name in ['total_phys','avail_phys','total_commit','avail_commit','total_virtual','avail_virtual','extended']]
    status=MemoryStatus();status.length=ctypes.sizeof(status)
    kernel.GlobalMemoryStatusEx.argtypes=[ctypes.POINTER(MemoryStatus)]
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error(),'Commit headroom query failed')
    return bool(low.value),int(status.avail_commit)


def memory_stop(peak_rss):
    """Keep a process cap; optionally let Windows manage physical paging.

    The OS policy never changes a system setting. It stops on the actual OS
    low-memory event, <1GiB commit headroom, or the unchanged 3.5GiB RSS cap.
    The historical default remains the conservative 650MiB physical reserve.
    """
    if peak_rss>3.5*2**30:return 'process RSS guard'
    if os.environ.get('COMMITJL_MEMORY_PRESSURE_POLICY')=='windows_os':
        try:low,headroom=_windows_memory_pressure()
        except (OSError,AttributeError):return 'RAM reserve'
        if low or headroom<1024*2**20:return 'RAM reserve'
        return None
    return 'RAM reserve' if psutil.virtual_memory().available<650*2**20 else None
