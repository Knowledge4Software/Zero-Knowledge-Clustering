"""Read live identities and complete checkpoint counts for the current review."""
from pathlib import Path
import json,time
import psutil
from campaigns.local_fast_tradeoff import save

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'code/validation/outputs/referee_two_reviews_20261027'


def main():
    live=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            args=p.info['cmdline'] or []
            if p.info['name'] not in ['python.exe','full-plonk-control-referee-20261027.exe','commitjl-bound-final-optimized-20261026.exe']:continue
            if not any('referee_two_reviews_20261027' in a for a in args):continue
            live.append(dict(pid=p.pid,name=p.name(),created_unix=p.create_time(),rss_bytes=p.memory_info().rss,command=args))
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    queue=json.loads((ROOT/'queue.json').read_text());follow=ROOT/'followups/queue.json'
    result=dict(updated_unix=time.time(),running=bool(live),active_processes=live,main=queue,
        followups=json.loads(follow.read_text()) if follow.exists() else None,maximum_proof_catalog_records=1024,mobin_retired=True,
        all_scientific_issues_fixed=False,author_metadata_deferred=True,public_release_authorized=False)
    save(REPO/'output/two_review_campaign_status_20261027.json',result)
    registry=REPO/'code/validation/outputs/research_jobs_20261024.json';old=json.loads(registry.read_text())
    old.update(updated_unix=time.time(),active_processes=live,current_status_file=(ROOT/'queue.json').relative_to(REPO).as_posix(),
        current_status='Two-review local revision: '+queue['status']+', phase '+str(queue.get('active_phase')),
        maximum_proof_catalog_records=1024,mobin_server_retired=True,mobin_connection_attempts_allowed=False)
    save(registry,old);print(json.dumps(dict(running=bool(live),processes=len(live),phase=queue.get('active_phase'))))


if __name__=='__main__':main()
