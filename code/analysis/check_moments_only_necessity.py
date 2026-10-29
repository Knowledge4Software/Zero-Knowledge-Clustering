"""Read-only numerical ablation requested by the 25 October referee.

This does not implement/time the authenticated moments-only baseline, generate
proofs, resume campaigns, or change the manuscript. All arithmetic is integer.
"""
from pathlib import Path
import hashlib
import itertools
import json


REPO=Path(__file__).resolve().parents[2]


def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def read(path):return json.loads(path.read_text())


def distance(row,center):return sum((a-b)**2 for a,b in zip(row,center))


def analyze(root):
    inp=read(root/'input.json');task=read(root/'approved.json')
    records=read(root/'bundle/group_query_records.json')
    checks=read(root/'native_checks.json')['decisions']
    rows=inp['x'];old=inp['model']['centers'];k=len(old);d=len(old[0])
    labels=[min(range(k),key=lambda g:distance(row,old[g])) for row in rows]
    counts=[labels.count(g) for g in range(k)]
    sums=[[sum(row[t] for i,row in enumerate(rows) if labels[i]==g) for t in range(d)] for g in range(k)]
    old_norm=[sum(v*v for v in center) for center in old]
    before=sum(distance(row,old[labels[i]]) for i,row in enumerate(rows))
    result=[]
    for index,new in enumerate(task['reuse_centers']):
        norms=[sum(v*v for v in center) for center in new]
        costs=[[2*sum(s*(o-v) for s,o,v in zip(sums[g],old[g],new[j]))+counts[g]*(norms[j]-old_norm[g])
                for j in range(k)] for g in range(k)]
        assigned=sum(costs[g][g] for g in range(k))
        assert assigned==int(records[index]['assigned_change'])
        best_permutation=min(sum(costs[g][p[g]] for g in range(k)) for p in itertools.permutations(range(k)))
        # Each old group can independently use its best new target. This is
        # an even stronger moments-only upper bound than a single permutation.
        best_groupwise=sum(min(row) for row in costs)
        new_costs=[min(distance(row,center) for center in new) for row in rows]
        assigned_new=[distance(row,new[labels[i]]) for i,row in enumerate(rows)]
        truth=sum(new_costs)-before
        assert truth==checks[index]['truth'] and truth<=best_groupwise<=best_permutation<=assigned
        result.append(dict(set=index,A=assigned,true_change=truth,
                           simple_A_certifies_improvement=assigned<0,
                           best_permutation_upper=best_permutation,
                           best_groupwise_upper=best_groupwise,
                           strict_switch_rows=sum(a<b for a,b in zip(new_costs,assigned_new)),
                           current_decision=checks[index]['decision']))
    files=['input.json','approved.json','bundle/group_query_records.json','native_checks.json']
    return dict(root=root.relative_to(REPO).as_posix(),updates=len(result),
                simple_A_improvements=sum(r['simple_A_certifies_improvement'] for r in result),
                best_permutation_improvements=sum(r['best_permutation_upper']<0 for r in result),
                best_groupwise_improvements=sum(r['best_groupwise_upper']<0 for r in result),
                current_conclusive=sum(r['current_decision']!='inconclusive' for r in result),
                actual_worsening_updates=sum(r['true_change']>0 for r in result),
                source_sha256={name:sha(root/name) for name in files},per_update=result)


def main():
    base=REPO/'code/validation/outputs'
    cases=[analyze(base/'group_compact_auxiliary_r79_n128_20261025/control'),
           analyze(base/'rank79_independent_replication_20261025/fashion_mnist_k3_seed101_age8/head79')]
    out=base/'referee_necessity_check_20261025';out.mkdir(exist_ok=True)
    report=dict(complete=True,scope='Numerical ablation of saved completed workloads; no new proofs/timings.',
                authenticated_moments_only_baseline_measured=False,main_campaigns_resumed=False,cases=cases)
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps([dict(root=c['root'],updates=c['updates'],simple_A_improvements=c['simple_A_improvements'],
                          best_groupwise_improvements=c['best_groupwise_improvements'],
                          current_conclusive=c['current_conclusive'],actual_worsening_updates=c['actual_worsening_updates'],
                          strict_switch_rows=[r['strict_switch_rows'] for r in c['per_update']]) for c in cases],indent=2))


if __name__=='__main__':main()
