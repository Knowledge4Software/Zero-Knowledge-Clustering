"""Bounded background supervisor for checkpointed laptop paper runs.

Resumes time-limited attempts, stops on worker errors, and never starts
concurrent benchmark workers. Closing the chat does not stop this process.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from campaigns.paper_compression import CODE,exclusive_run
from experiments.laptop_comparison import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--profile',choices=['balanced','paper'],default='balanced')
    parser.add_argument('--max-hours',type=float,default=30)
    parser.add_argument('--max-attempts',type=int,default=10)
    args=parser.parse_args()
    if args.max_hours<=0 or args.max_attempts<1:parser.error('Positive budgets required')
    root=args.out.resolve();control=root/'background';control.mkdir(parents=True,exist_ok=True)
    def count():return sum(1 for _ in (root/'jobs').glob('*/rows/*.json'))
    def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    with exclusive_run(control):
        began=time.monotonic();deadline=began+args.max_hours*3600
        status=dict(start_utc=utc(),profile=args.profile,max_hours=args.max_hours,
            max_attempts=args.max_attempts,status='starting',attempts=[],pid=__import__('os').getpid())
        state_path=control/'status.json'
        def save():
            status.update(updated_utc=utc(),elapsed_seconds=time.monotonic()-began,completed_rows=count())
            atomic_json(state_path,status)
        save();stalled=0
        try:
            for index in range(args.max_attempts):
                remaining=deadline-time.monotonic()
                if remaining<60:status['status']='budget_exhausted';break
                before=count();status['status']='running';save()
                cmd=[sys.executable,'-u','-m','campaigns.paper_compression',
                    '--profile',args.profile,'--out',str(root),'--run',
                    '--max-minutes',str(min(480.,remaining/60)), '--max-rss-gib','3']
                print(json.dumps(dict(event='attempt_started',attempt=index+1,utc=utc(),rows=before)),flush=True)
                result=subprocess.run(cmd,cwd=CODE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                manifest_path=root/'manifest.json'
                manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
                after=count();status['attempts'].append(dict(attempt=index+1,returncode=result.returncode,rows_before=before,rows_after=after,ended_utc=utc()))
                if result.returncode:
                    status.update(status='error',reason='Campaign failed; inspect stdout/stderr. Automatic retries stopped.');break
                if manifest.get('complete'):
                    status['status']='complete';break
                records=manifest.get('attempts',[])
                last_jobs=records[-1].get('jobs',[]) if records else []
                if any(j.get('status')=='error' for j in last_jobs):
                    status.update(status='error',reason='Worker error; automatic retries stopped.');break
                stalled=stalled+1 if after<=before else 0
                if stalled>=2:
                    status.update(status='stalled',reason='Two attempts made no checkpoint progress; automatic retries stopped.');break
                status['status']='waiting_to_resume';save()
                print(json.dumps(dict(event='checkpoint_resume',attempt=index+1,rows=after)),flush=True)
                if deadline-time.monotonic()>60:time.sleep(60)
            else:status['status']='attempt_budget_exhausted'
        except Exception as exc:
            status.update(status='error',reason=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            save();print(json.dumps(dict(event='supervisor_finished',status=status['status'],rows=status['completed_rows'])),flush=True)


if __name__=='__main__':main()
