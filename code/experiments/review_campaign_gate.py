"""Wait for the preceding numerical queue to stop, without benchmark contention."""
import argparse,json,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--queue',type=Path,required=True);p.add_argument('--state',type=Path,required=True)
    a=p.parse_args();start=time.time()
    while True:
        log=a.queue.read_text() if a.queue.exists() else ''
        terminal='ALL_DONE' in log or 'FAILED_JOBS=' in log
        state=dict(status='ready' if terminal else 'waiting_for_previous_numerical_queue',elapsed_seconds=time.time()-start,queue=str(a.queue))
        a.state.write_text(json.dumps(state,indent=2))
        if terminal:return
        if time.time()-start>24*3600:raise TimeoutError('prior numerical queue did not terminate within 24 hours')
        time.sleep(10)


if __name__=='__main__':main()
