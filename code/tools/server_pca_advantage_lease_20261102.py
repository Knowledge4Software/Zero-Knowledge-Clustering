"""Yield an existing coordinator at a phase boundary and always resume it.

No proof worker in the older campaign is paused or terminated. New study
workers are stopped before resuming the older coordinator on any exit path.
"""
from pathlib import Path
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time

import psutil

from experiments.pca_advantage_20261102 import read, save


def matches(process, pin):
    try:
        return (process.pid == pin['pid'] and abs(process.create_time()-pin['created']) < .01
                and pin['module'] in process.cmdline())
    except psutil.Error:
        return False


def study_processes(root, exclude=()):
    prefix = str(Path(root).resolve())
    found = []
    for process in psutil.process_iter(['pid', 'cmdline', 'create_time', 'status']):
        if process.pid in exclude or process.info['status'] == psutil.STATUS_ZOMBIE:
            continue
        args = process.info['cmdline'] or []
        if any(arg == prefix or arg.startswith(prefix+'/') for arg in args):
            found.append(process)
    return found


def stop_study(root, exclude):
    """Only processes whose arguments name this isolated output directory."""
    processes = study_processes(root, exclude)
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, live = psutil.wait_procs(processes, timeout=12)
    for process in live:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(live, timeout=8)
    if study_processes(root, exclude):
        raise RuntimeError('New-study workers remain; do not overlap proof timing')


def resume_legacy(state):
    pin = state.get('legacy')
    if not pin:
        return False
    try:
        process = psutil.Process(pin['pid'])
        if matches(process, pin):
            process.send_signal(signal.SIGCONT)
            return True
    except psutil.NoSuchProcess:
        pass
    return False


def watchdog(path):
    """Independent owner-death/deadline recovery; survives laptop disconnects."""
    path = Path(path)
    while True:
        state = read(path)
        if state.get('complete'):
            return
        pin = state['owner']
        try:
            owner_live = matches(psutil.Process(pin['pid']), pin)
        except psutil.NoSuchProcess:
            owner_live = False
        if not owner_live or time.time() >= state['deadline_unix']:
            (path.parent/'STOP').write_text('Owned study deadline/owner-death stop.\n')
            stop_study(path.parent, [os.getpid()])
            state.update(complete=True, status='watchdog_recovered',
                         finished_unix=time.time(), legacy_resumed=resume_legacy(state))
            save(path, state)
            return
        time.sleep(3)


class LegacyLease:
    def __init__(self, root, legacy_pid=24249, hours=4):
        self.root = Path(root).resolve()
        self.path = self.root/'lease.json'
        self.legacy_pid = legacy_pid
        self.hours = hours
        self.watch = None
        self.lock = None
        self.state = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise RuntimeError('Lease already exists; recovery needs a separate recorded action')
        owner = psutil.Process()
        self.state = dict(complete=False, status='acquiring', started_unix=time.time(),
             owner=dict(pid=owner.pid, created=owner.create_time(),
                        module='campaigns.server_pca_advantage_20261102'),
             deadline_unix=time.time()+(self.hours+1.5)*3600,
             legacy=None, interpretation='Old coordinator only is stopped; its current proof phase drains normally. Coordinator/cell elapsed may include this queue wait; retain component proof timings.')
        try:
            process = psutil.Process(self.legacy_pid)
            module = 'campaigns.server_all_sources_20261101'
            if module not in process.cmdline():
                raise RuntimeError('Legacy PID is not the expected coordinator')
            self.state['legacy'] = dict(pid=process.pid, created=process.create_time(), module=module)
        except psutil.NoSuchProcess:
            primary = self.root.parent/'server_campaign_20261101/coordinator.lock'
            self.lock = primary.open('a')
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(self.path, self.state)
        self.watch = subprocess.Popen([sys.executable, '-u', '-m',
             'tools.server_pca_advantage_lease_20261102', '--watchdog', str(self.path)],
             start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.state['watchdog_pid'] = self.watch.pid
        save(self.path, self.state)
        try:
            if self.state['legacy']:
                process = psutil.Process(self.legacy_pid)
                if not matches(process, self.state['legacy']):
                    raise RuntimeError('Legacy identity changed before yielding')
                process.send_signal(signal.SIGSTOP)
                self.state.update(status='draining_legacy_phase', paused_unix=time.time())
                save(self.path, self.state)
                while True:
                    children = [p for p in process.children(recursive=True)
                                if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]
                    if not children:
                        break
                    if time.time() >= self.state['deadline_unix']:
                        raise RuntimeError('Legacy drain exceeded the lease deadline')
                    time.sleep(1)
            self.state.update(status='exclusive_study', exclusive_started_unix=time.time(),
                              deadline_unix=time.time()+self.hours*3600)
            save(self.path, self.state)
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, kind, value, traceback):
        if self.state is not None:
            exclude = [os.getpid()] + ([self.watch.pid] if self.watch else [])
            stop_study(self.root, exclude)
            self.state.update(complete=True, status='released', finished_unix=time.time(),
                              error=str(value) if value else None,
                              legacy_resumed=resume_legacy(self.state))
            save(self.path, self.state)
        if self.watch:
            try:
                self.watch.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.watch.terminate()
                self.watch.wait(timeout=5)
        if self.lock:
            self.lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watchdog', type=Path, required=True)
    watchdog(parser.parse_args().watchdog)
