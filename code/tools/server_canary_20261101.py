"""Standard-library-only largest-catalog real-proof canary during provisioning.

Uses frozen previously inspected inputs. This is resource calibration, not an
independent-fit comparative result. It never requires Python ML dependencies.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def memory():
    return {line.split(':')[0]:int(line.split()[1])*1024
            for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith(('MemTotal:', 'MemAvailable:'))}


def process_usage(pid):
    # The adapter is one multi-threaded Rust process, with no child processes.
    status = Path(f'/proc/{pid}/status').read_text()
    values = {line.split(':')[0]:line.split(':',1)[1].strip() for line in status.splitlines() if ':' in line}
    stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return dict(rss_bytes=int(values.get('VmRSS','0 kB').split()[0])*1024,
                high_water_bytes=int(values.get('VmHWM','0 kB').split()[0])*1024,
                threads=int(values.get('Threads','0')),
                cpu_seconds=(int(stat[11])+int(stat[12]))/os.sysconf('SC_CLK_TCK'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    profile = read(Path(__file__).resolve().parents[2]/'server_profile.json')
    assert os.uname().nodename == profile['hostname']
    affinity = sorted(os.sched_getaffinity(0))[:profile['workers']]
    assert len(affinity) == profile['workers']
    os.sched_setaffinity(0, affinity)
    env = dict(os.environ, RAYON_NUM_THREADS=str(profile['workers']),
               GOMAXPROCS=str(profile['workers']), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    state = dict(status='running', started_unix=started, pid=os.getpid(), workers=profile['workers'],
                 binary_sha256=digest(args.binary), source_input_sha256=digest(args.input),
                 scope='Fresh real proofs on frozen historical fit; engineering resource calibration only.', phases=[])
    save(root/'state.json', state)

    def run(label, command, expected_failure=False):
        receipt = root/'resources'/(label+'.json')
        if receipt.exists():
            raise RuntimeError('Canary attempts are immutable; choose a new directory')
        receipt.parent.mkdir(exist_ok=True)
        begin = time.time()
        peak = 0
        usage = dict(cpu_seconds=0, rss_bytes=0, high_water_bytes=0, threads=0)
        stop = None
        progress = dict(label=label, command=list(map(str, command)),
                        started_unix=begin, peak_rss_bytes=0, workers=profile['workers'])
        with receipt.with_suffix('.log').open('wb') as log:
            child = subprocess.Popen(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
            state.update(active_phase=label, child_pid=child.pid)
            save(root/'state.json', state)
            while child.poll() is None:
                try:
                    usage = process_usage(child.pid)
                    peak = max(peak, usage['rss_bytes'], usage['high_water_bytes'])
                except (FileNotFoundError, ProcessLookupError):
                    pass
                elapsed = time.time()-begin
                if peak > profile['rss_cap_gib']*2**30:
                    stop = 'RSS cap'
                elif memory()['MemAvailable'] < profile['reserve_ram_gib']*2**30:
                    stop = 'RAM reserve'
                elif shutil.disk_usage(root).free < profile['reserve_disk_gib']*2**30:
                    stop = 'disk reserve'
                elif elapsed > 5400 or (root/'STOP').exists():
                    stop = 'timeout or STOP'
                progress = dict(complete=False, running=True, label=label, command=list(map(str, command)),
                    started_unix=begin, wall_seconds=elapsed, peak_rss_bytes=peak, workers=profile['workers'],
                    average_cpu_cores=usage['cpu_seconds']/max(elapsed,.001), **usage)
                save(receipt, progress)
                if stop:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid, signal.SIGKILL);child.wait()
                    break
                time.sleep(1)
        code = child.wait()
        progress.update(complete=(code!=0 if expected_failure else code==0) and stop is None,
                        running=False, wall_seconds=time.time()-begin, returncode=code,
                        stop=stop, expected_failure=expected_failure)
        save(receipt, progress)
        state['phases'].append(progress)
        save(root/'state.json', state)
        if not progress['complete']:
            raise RuntimeError(f'{label}: exit{code}, {stop}')

    try:
        smoke = root/'smoke'
        model = dict(n=4,d=3,k=2,input_bits=4)
        small = dict(model=model,x=[[0,0,1],[4,0,1],[7,7,2],[1,7,2]],salt_x='11',
                     reuse_centers=[[[0,0,0],[8,8,0]],[[1,0,0],[7,8,0]]])
        save(smoke/'input.json', small)
        save(smoke/'approved.json', dict(reuse_centers=small['reuse_centers']))
        run('smoke_prepare',[args.binary,'prepare',smoke/'input.json',smoke/'private',12,17])
        run('smoke_false_witness_checks',[args.binary,'selftest',smoke/'input.json',smoke/'private'])
        run('smoke_query',[args.binary,'query',smoke/'input.json',smoke/'private',1])
        public = smoke/'public';public.mkdir()
        for name in ['params','vk']:shutil.copy2(smoke/'private'/name,public/name)
        (public/'bank_001').mkdir()
        for name in ['proof','public.json']:shutil.copy2(smoke/'private/bank_001'/name,public/'bank_001'/name)
        verify = [args.binary,'verify',public,1,smoke/'private/owner_pin.json',smoke/'approved.json']
        run('smoke_verify',verify)
        proof = public/'bank_001/proof';original=proof.read_bytes()
        proof.write_bytes(bytes([original[0]^1])+original[1:])
        try:run('smoke_bad_proof',verify,True)
        finally:proof.write_bytes(original)
        original = read(args.input)
        for n in [1024,512]:
            case=root/f'fashion_n{n}'
            inp=copy.deepcopy(original);inp['x']=inp['x'][:n];inp['model']['n']=n
            inp['reuse_centers']=inp['reuse_centers'][:2]
            inp['queries']=[list(range(i,i+8)) for i in range(0,n,8)]
            save(case/'input.json',inp);save(case/'approved.json',dict(reuse_centers=inp['reuse_centers']))
            label=f'n{n}_inspect_k20'
            try:run(label,[args.binary,'inspect',case/'input.json',case/'inspection',20,17])
            except RuntimeError:
                text=(root/'resources'/(label+'.log')).read_text(errors='replace')
                if 'NotEnoughRowsAvailable' not in text and 'usable_rows=' not in text:raise
                save(case/'excluded.json',dict(reason='Does not fit the pinned degree20 layout',input_sha256=digest(case/'input.json')))
                continue
            state['selected_n']=n;save(root/'state.json',state)
            private=case/'private';public=case/'public';public.mkdir()
            run(f'n{n}_prepare',[args.binary,'prepare',case/'input.json',private,20,17])
            for name in ['params','vk']:os.link(private/name,public/name)
            run(f'n{n}_queries',[args.binary,'queries',case/'input.json',private])
            costs=[]
            for bank,centers in enumerate(inp['reuse_centers']):
                dest=public/f'bank_{bank:03}';dest.mkdir()
                for name in ['proof','public.json']:shutil.copy2(private/f'bank_{bank:03}'/name,dest/name)
                run(f'n{n}_verify{bank}',[args.binary,'verify',public,bank,private/'owner_pin.json',case/'approved.json'])
                exact=sum(min(sum((x-c)**2 for x,c in zip(row,center)) for center in centers) for row in inp['x'])
                assert exact==read(private/f'bank_{bank:03}/result.json')['cost']
                costs.append(exact)
            save(case/'checked.json',dict(complete=True,costs=costs,all_costs_match_integer_oracle=True))
            break
        else:raise RuntimeError('Neither catalog fits the pinned adapter')
        state.update(status='complete',complete=True,finished_unix=time.time(),active_phase=None)
    except BaseException as error:
        state.update(status='stopped_with_preserved_evidence',complete=False,error=str(error),finished_unix=time.time())
        raise
    finally:
        save(root/'state.json',state)


if __name__=='__main__':
    main()
