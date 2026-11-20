"""Monitor an isolated server phase, retaining CPU/RAM/disk and failed attempts."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil
from campaigns.proof_runtime import server_profile
from commitjl.pilot_io import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=5400)
    parser.add_argument('--expect-failure', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    profile = server_profile()
    if not profile:
        raise RuntimeError('An authorized server resource profile is required')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.receipt.exists():
        raise ValueError('Missing command or an attempt receipt already exists')
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    affinity = sorted(os.sched_getaffinity(0))[:profile['workers']]
    os.sched_setaffinity(0, affinity)
    started = time.time()
    peak = 0
    stop = None
    progress = dict(command=command, started_unix=started, workers=profile['workers'],
                    cpu_affinity=affinity, peak_process_tree_rss_bytes=0,
                    resource_profile=profile)
    received_signal = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda number, frame: received_signal.append(number))
    with args.log.open('wb') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        process = psutil.Process(child.pid)
        last_save = 0
        while child.poll() is None:
            rss = 0
            cpu = 0
            try:
                for item in [process] + process.children(recursive=True):
                    try:
                        rss += item.memory_info().rss
                        times = item.cpu_times()
                        cpu += times.user + times.system
                    except psutil.Error:
                        pass
            except psutil.Error:
                pass
            peak = max(peak, rss)
            elapsed = time.time() - started
            available = psutil.virtual_memory().available
            free_disk = psutil.disk_usage(args.root).free
            if rss > profile['rss_cap_gib']*2**30:
                stop = 'process-tree RSS cap'
            elif available < profile['reserve_ram_gib']*2**30:
                stop = 'host RAM reserve'
            elif free_disk < profile['reserve_disk_gib']*2**30:
                stop = 'disk reserve'
            elif elapsed > args.timeout:
                stop = 'phase timeout'
            elif (args.root/'STOP').exists() or received_signal:
                stop = 'stop requested'
            progress = dict(complete=False, running=True, pid=child.pid,
                            command=command, started_unix=started,
                            wall_seconds=elapsed, peak_process_tree_rss_bytes=peak,
                            current_process_tree_rss_bytes=rss, cpu_seconds=cpu,
                            average_cpu_cores=cpu/max(elapsed, .001),
                            available_ram_bytes=available, free_disk_bytes=free_disk,
                            workers=profile['workers'], cpu_affinity=affinity,
                            resource_profile=profile, stop=stop)
            if elapsed-last_save >= 5:
                save(args.receipt, progress)
                last_save = elapsed
            if stop:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                except ProcessLookupError:
                    pass
                break
            time.sleep(.5)
    code = child.wait()
    passed = (code != 0 if args.expect_failure else code == 0) and stop is None
    progress.update(complete=passed, running=False, returncode=code,
                    wall_seconds=time.time()-started, stop=stop,
                    expected_failure=args.expect_failure)
    save(args.receipt, progress)
    print(json.dumps(progress), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
