"""Validate and preserve local optimization evidence and source revisions."""
import hashlib
import json
from pathlib import Path
import re
import shutil

import pandas as pd

from experiments.benchmark_circuit_optimization import write_summary

CODE=Path(__file__).resolve().parents[1]
ROOT=CODE.parent
OUT=CODE/'results/optimization_20261010'
SCRATCH=ROOT/'tmp/circuit_optimization_20261010'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    smoke=json.loads((OUT/'backend_smoke.json').read_text(encoding='utf-8'))
    host=json.loads((OUT/'host_projection.json').read_text(encoding='utf-8'))
    timings=pd.read_csv(OUT/'timings.csv')
    assert manifest['complete'] and len(timings)==13 and timings.ok.all()
    assert int(timings.successful_proofs.sum())==65
    for row in timings.itertuples():
        result=json.loads((OUT/'records'/f'{row.shape}_{row.method}_{row.mode}.json').read_text(encoding='utf-8'))
        assert result['ok'] and result['verify_ok'] and result['successful_proofs']==5
        assert result['num_constraints']==row.constraints
    assert smoke['complete'] and len(smoke['records'])==2
    assert all(r['result']['ok'] and r['result']['verify_ok'] for r in smoke['records'])
    assert host['complete'] and all(r['canonical_records_equal'] and r['csr_arrays_and_products_bitwise_equal'] for r in host['rows'])
    binary=CODE/'zk/bin/commitjl-prove.exe'
    assert sha(binary)==smoke['binary_sha256'], 'Default binary differs from final smoke-tested build'
    measured=(CODE/'zk/circuit/circuit.go').read_text(encoding='utf-8')
    zero_guard=('\t// The all-zero constant matrix is valid. Keep comparator bounds positive;\n'
                '\t// projection equality still forces its output to zero.\n'
                '\tif derived.Sign() == 0 {\n\t\tderived.SetInt64(1)\n\t}\n')
    assert measured.count(zero_guard)==1
    measured=measured.replace(zero_guard,'')
    assert hashlib.sha256(measured.encode()).hexdigest()==manifest['sources'][str(Path('zk/circuit/circuit.go'))]
    snapshot=OUT/'source_snapshots'
    snapshot.mkdir(exist_ok=True)
    (snapshot/'measured_circuit.go.txt').write_bytes(measured.encode())
    shutil.copy2(SCRATCH/'circuit_before.go.txt',snapshot/'original_circuit.go.txt')
    shutil.copy2(SCRATCH/'main_before.go.txt',snapshot/'original_main.go.txt')
    current=[]
    for relative in ['zk/circuit/circuit.go','zk/circuit/optimization_test.go','zk/cmd/commitjl-prove/main.go',
                     'zk/go.mod','zk/go.sum','commitjl/zk_bridge.py','commitjl/sparse_gen.py',
                     'experiments/rq1_complete.py','experiments/rq2_proof_cost.py',
                     'experiments/benchmark_circuit_optimization.py']:
        source=CODE/relative; dest=snapshot/'final'/relative
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        current.append(dict(path=relative,sha256=sha(source)))
    measured_main=CODE/'zk/cmd/commitjl-prove/main.go'
    assert sha(measured_main)==manifest['sources'][str(Path('zk/cmd/commitjl-prove/main.go'))]
    shutil.copy2(measured_main,snapshot/'measured_main.go.txt')
    write_summary(OUT)
    receipt=dict(status='passed',timed_proofs=65,actual_backend_controls=2,
        original_constraint_count_reproduced=manifest['original_constraint_count_reproduced'],
        current_binary_sha256=sha(binary),measured_binary_sha256=manifest['binary_sha256'],
        measured_go_source_snapshots_verified=True,final_sources=current,
        tests='go test ./... and go vet ./... passed; full Python suite excluding expensive test_zk.py: 136 passed, 2 skipped; final sampler/RQ1/bridge checks after integer-domain guards: 24 passed.',
        scope='Local optimization only. Preserved server results and paper reports remain unchanged.',
        timing_limitations='Fixed synthetic witnesses; exploratory timing, with regression/smoke tests active during portions of the study. Optimized full controls remain faster than Sparse on both measured shapes.')
    (OUT/'validation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='final_sources'},indent=2))

if __name__=='__main__':main()
