"""Finite, unattended Ubuntu queue for the balanced paper campaigns.

Run under systemd. --status only reads state; it never interrupts experiments.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def build_plan(code, root, python):
    code, root = Path(code).resolve(), Path(root).resolve()
    rq1 = root / 'rq1'
    common = ['--run', '--seeds', '10', '--starts', '5', '--jobs', '4',
              '--threads', '4', '--proof-threads', '16', '--proof-rss-gib', '40',
              '--task-rss-gib', '48', '--bootstrap', '2000', '--rq1', str(rq1)]
    def suite(name, stages, hours, reps=10):
        return dict(name=name, command=[python, '-u', '-m', 'experiments.rq2345_complete',
                    *common, '--stages', *stages, '--proof-reps', str(reps),
                    '--out', str(root/name)], timeout_seconds=hours*3600, requires=[])
    plan = [dict(name='rq1', command=[python, '-u', '-m', 'experiments.rq1_complete',
                '--run', '--seeds', '10', '--starts', '5', '--jobs', '4', '--threads', '4',
                '--max-rss-gib', '8', '--dataset-timeout-hours', '4', '--out', str(rq1)],
                timeout_seconds=8*3600, requires=[]),
            dict(name='rq1_export', command=[python, '-u', '-m', 'analysis.rq1_export',
                 '--runs', str(rq1)], timeout_seconds=3600, requires=['rq1']),
            suite('native', ['rq3_native', 'rq3_grinding', 'rq4_centroid', 'rq4_dbscan',
                            'rq4_phase', 'rq5_native', 'rq5_precision'], 8),
            suite('selected', ['rq2_quality', 'rq2_selected'], 4),
            suite('pilot', ['rq2_pilot'], 24, reps=3),
            suite('proof', ['rq3_proof', 'rq5_proof', 'rq5_ablation'], 24)]
    for phase in plan:
        if phase['name'] in ['native', 'selected']:
            phase['requires'] = ['rq1_export']
    return plan


def stop_tree(process):
    """Stop the phase and its children before proceeding to another phase."""
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.Error:
        children = []
    if os.name == 'posix':
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    for child in children:
        try:
            child.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(children, timeout=5)


def run_bounded(command, cwd, log, timeout_seconds, heartbeat=lambda _: None,
                poll_seconds=5, idle_seconds=1200, rss_gib=49, min_free_gib=10):
    start = time.monotonic()
    last_activity = start
    cpu_seen = {}
    peak = 0
    last_log_size = 0
    reason = None
    with Path(log).open('a', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=cwd, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                now = time.monotonic()
                rss = 0
                cpu_delta = 0
                try:
                    parent = psutil.Process(process.pid)
                    for item in [parent] + parent.children(recursive=True):
                        try:
                            rss += item.memory_info().rss
                            times = item.cpu_times()
                            total = times.user + times.system
                            cpu_delta += max(0, total-cpu_seen.get(item.pid, 0))
                            cpu_seen[item.pid] = total
                        except psutil.Error:
                            pass
                except psutil.Error:
                    pass
                peak = max(peak, rss)
                size = Path(log).stat().st_size
                if cpu_delta >= .01 or size != last_log_size:
                    last_activity = now
                last_log_size = size
                elapsed = now-start
                if elapsed >= timeout_seconds:
                    reason = 'phase_timeout'
                elif now-last_activity >= idle_seconds:
                    reason = 'no_cpu_or_log_progress'
                elif rss > rss_gib*2**30:
                    reason = 'queue_memory_limit'
                elif psutil.disk_usage(str(cwd)).free < min_free_gib*2**30:
                    reason = 'disk_space_limit'
                heartbeat(dict(child_pid=process.pid, elapsed_seconds=elapsed,
                               peak_rss_gib=peak/2**30, rss_gib=rss/2**30,
                               idle_seconds=now-last_activity, updated_at_utc=utc_now()))
                if reason:
                    stop_tree(process)
                    break
                time.sleep(min(poll_seconds, max(.01, timeout_seconds-elapsed)))
        except BaseException:
            stop_tree(process)
            raise
    return dict(exit_code=process.wait(), reason=reason,
                elapsed_seconds=time.monotonic()-start, peak_rss_gib=peak/2**30,
                finished_at_utc=utc_now())


def run_queue(code, root, plan):
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'.queue.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_path = root/'queue_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Queue commands changed; use a fresh results directory')
    atomic_json(plan_path, plan)
    state_path = root/'queue_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(
        started_at_utc=utc_now(), phases={}, root=str(root))
    state.update(status='running', controller_pid=os.getpid(), updated_at_utc=utc_now())
    atomic_json(state_path, state)
    for phase in plan:
        name = phase['name']
        previous = state['phases'].get(name, {})
        if previous.get('status') in ['completed', 'failed', 'skipped']:
            continue
        unmet = [dep for dep in phase['requires']
                 if state['phases'].get(dep, {}).get('status') != 'completed']
        if unmet:
            state['phases'][name] = dict(status='skipped', reason='dependency_failed',
                                          dependencies=unmet, finished_at_utc=utc_now())
            atomic_json(state_path, state)
            continue
        record = dict(status='running', started_at_utc=utc_now(),
                      command=phase['command'], timeout_seconds=phase['timeout_seconds'],
                      log=str(root/(name+'.log')))
        state['phases'][name] = record
        state.update(current_phase=name, updated_at_utc=utc_now())
        atomic_json(state_path, state)
        print(json.dumps(dict(event='phase_started', phase=name, time=utc_now())), flush=True)
        def heartbeat(update):
            record.update(update)
            state['updated_at_utc'] = update['updated_at_utc']
            atomic_json(state_path, state)
        outcome = run_bounded(phase['command'], code, record['log'],
                              phase['timeout_seconds'], heartbeat=heartbeat)
        record.update(outcome, status='completed' if outcome['exit_code']==0 and
                      outcome['reason'] is None else 'failed')
        atomic_json(state_path, state)
        print(json.dumps(dict(event='phase_finished', phase=name, outcome=record)), flush=True)
    failures = [name for name, row in state['phases'].items() if row['status'] != 'completed']
    state.update(status='finished_with_failures' if failures else 'completed',
                 current_phase=None, finished_at_utc=utc_now(), updated_at_utc=utc_now(),
                 failed_or_skipped_phases=failures)
    atomic_json(state_path, state)
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--status', action='store_true')
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[1]
    root = args.root.resolve()
    if args.status:
        path = root/'queue_state.json'
        print(path.read_text() if path.exists() else json.dumps({'status':'not_started'}))
        return 0
    plan = build_plan(code, root, sys.executable)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    return run_queue(code, root, plan)


if __name__ == '__main__':
    raise SystemExit(main())
