"""Reconstruct matched public-center Full/plain/certified costs in a NEW export.

Only completed captured cases are admitted. Check transfer hashes, requested
coverage, integer Full outputs and independent certificate outputs. Each
deployment alternative pays common preparation once; the experimental plain
phase is not charged to the certified alternative.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
sys.path.insert(0, str(CODE))
import numpy as np
from commitjl.reuse_certificates import certificate


def load(path):
    return json.loads(path.read_text())


def checked_capture(path):
    manifest = load(path/'capture.json')
    for record in manifest['files']:
        assert hashlib.sha256((path/record['path']).read_bytes()).hexdigest() == record['sha256']
    return manifest


def phase_cost(phase):
    return phase['compile_seconds'] + phase['setup_seconds'] + sum(
        x.get('native_answer_seconds', 0.) + x['witness_seconds'] + statistics.median(x['prove_seconds'])
        for x in phase.get('samples') or [])


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-capture', type=Path, required=True)
    p.add_argument('--reuse-capture', type=Path, required=True)
    p.add_argument('--additional-capture', type=Path, action='append', default=[],
                   help='Additional completed captures; Full and reuse cases may coexist.')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    capture_paths = list(dict.fromkeys([a.full_capture, a.reuse_capture, *a.additional_capture]))
    captures = [checked_capture(x) for x in capture_paths]
    cases = [p.parent for source in capture_paths for p in sorted(source.glob('*/summary.json'))]
    assert len({p.name for p in cases}) == len(cases), 'duplicate case names across captures'
    owner_dirs = [CODE/'validation/outputs/review_owner_inputs_20261020',
                  CODE/'validation/outputs/review_certificate_preflight_20261020/owner_inputs',
                  CODE/'validation/outputs/review_certificate_budget_20261020/owner_inputs']
    owners = {hashlib.sha256(p.read_bytes()).hexdigest(): p
              for d in owner_dirs for p in d.glob('*.json')}
    summaries = []; setrows = []; exact_by_root = {}; full_by_root = {}

    def inspect(case, kind):
        protocol = load(case/'protocol.json'); ownerpath = owners[protocol['source_sha256']]
        owner = load(ownerpath); n = protocol['n']; sets = protocol['center_sets']
        root = load(case/'summary.json')['original_root']; task = load(case/'approved_public_task.json')
        assert task['reuse_centers'] == owner['reuse_centers'][:sets]
        assert task['queries'] == [list(range(s,s+protocol['batch'])) for s in range(0,n,protocol['batch'])]
        xi = np.asarray(owner['x'],dtype=np.int64)
        exact = []
        for centers in task['reuse_centers']:
            distances = ((xi[:,None,:]-np.asarray(centers,dtype=np.int64)[None,:,:])**2).sum(2)
            labels = distances.argmin(1)
            exact.append((labels,int(distances[np.arange(n),labels].sum())))
        replay = load(case/'public_bundle/verify.json')
        assert replay['complete'] and load(case/'status.json')['complete']
        prep = load(case/'bundle/prepare.json'); phase = load(case/f"bundle/{kind}_{protocol['batch']}.json")
        assert prep['statement']['cx'] == root and phase['statement'] == prep['statement']
        assert phase['complete'] and phase['verified'] == len(phase['samples'])
        for s in range(sets):
            samples = sorted([x for x in phase['samples'] if x.get('centre_set',0)==s],key=lambda x:x['id'])
            assert [i for x in samples for i in x['indices']] == list(range(n))
        return protocol, ownerpath, owner, root, exact, replay, prep, phase

    for case in cases:
        if not list((case/'bundle').glob('full_reuse_*.json')):
            continue
        protocol, _, owner, root, exact, replay, prep, phase = inspect(case,'full_reuse')
        assert replay['verified'] == len(phase['samples'])
        for s,(labels,cost) in enumerate(exact):
            samples=sorted([x for x in phase['samples'] if x.get('centre_set',0)==s],key=lambda x:x['id'])
            assert [i for x in samples for i in x['assign']] == labels.tolist()
            assert sum(int(x['psi']) for x in samples) == cost
        full_by_root[root] = dict(case=case.name,total_seconds=prep['native_prepare_seconds']+phase_cost(phase),
                                  constraints=phase['constraints'],n=protocol['n'],batch=protocol['batch'],
                                  proofs=replay['verified'],peak_rss_bytes=load(case/'summary.json')['peak_rss_bytes'])
        exact_by_root[root] = (protocol['source_sha256'],owner['reuse_centers'],owner['x'])

    for case in cases:
        if not list((case/'bundle').glob('reuse_certified_*.json')):
            continue
        protocol, ownerpath, owner, root, exact, replay, prep, plain = inspect(case,'reuse')
        if root not in full_by_root:
            continue
        original = exact_by_root[root]
        assert owner['reuse_centers'] == original[1] and owner['x'] == original[2]
        stage1 = load(case/'bundle/stage1.json')
        cert = load(case/f"bundle/reuse_certified_{protocol['batch']}.json")
        assert stage1['complete'] and stage1['verified']==protocol['n'] and len(stage1['samples'])==protocol['n']
        assert stage1['statement'] == plain['statement']
        assert cert['complete'] and cert['statement']==plain['statement']
        assert replay['verified']==protocol['n']+len(plain['samples'])+len(cert['samples'])
        shared = prep['native_prepare_seconds']+phase_cost(stage1)
        method = 'Learned + energy' if 'learned' in case.name else 'PCA + energy'
        per_set = []
        for s,(labels,cost) in enumerate(exact):
            ps=sorted([x for x in plain['samples'] if x.get('centre_set',0)==s],key=lambda x:x['id'])
            cs=sorted([x for x in cert['samples'] if x.get('centre_set',0)==s],key=lambda x:x['id'])
            assert [i for x in cs for i in x['indices']] == list(range(protocol['n']))
            predicted=[i for x in ps for i in x['assign']]
            assert predicted == [i for x in cs for i in x['assign']]
            psi=sum(int(x['psi']) for x in ps)
            assert psi == sum(int(x['psi']) for x in cs)
            ref=certificate(owner['model'],owner['x'],owner['reuse_centers'][s])
            assert predicted == ref['labels'] and psi == ref['psi']
            lo=sum(int(x['cost_lower_scaled']) for x in cs); hi=sum(int(x['cost_upper_scaled']) for x in cs)
            count=sum(x.get('certified_count',0) for x in cs); flags=[i for x in cs for i in x['certified_flags']]
            assert (lo,hi,count,flags)==(ref['cost_lower_scaled'],ref['cost_upper_scaled'],ref['certified_count'],ref['certified_flags'])
            assert lo <= cost*owner['model']['basis_scale']**6 <= hi
            assert all(not flag or label==truth for flag,label,truth in zip(flags,predicted,labels))
            per_set.append(dict(case=case.name,set=s,n=protocol['n'],agreement_quantized=float(np.mean(np.asarray(predicted)==labels)),
                                certified_fraction=count/protocol['n'],interval_relative_width=(hi-lo)/lo if lo else None,
                                certified_count=count,lower=str(lo),upper=str(hi),
                                coverage99_cost1_accept=100*count>=99*protocol['n'] and lo>0 and 100*(hi-lo)<=lo))
        setrows += per_set
        for kind,phase in [('plain',plain),('certified',cert)]:
            total=shared+phase_cost(phase)
            summary = dict(case=case.name,method=method,mode=kind,n=protocol['n'],center_sets=protocol['center_sets'],
                dataset='CIFAR/ResNet' if case.name.startswith('cifar') else 'MNIST',
                rank=len(owner['model']['basis']),basis_scale=owner['model']['basis_scale'],batch=protocol['batch'],
                original_root=root,source_sha256=protocol['source_sha256'],source_local=ownerpath.relative_to(ROOT).as_posix(),
                shared_preparation_seconds=shared,query_compile_setup_seconds=phase['compile_seconds']+phase['setup_seconds'],
                native_answer_witness_prove_seconds=phase_cost(phase)-phase['compile_seconds']-phase['setup_seconds'],total_seconds=total,
                full_public_seconds=full_by_root[root]['total_seconds'],full_over_method_ratio=full_by_root[root]['total_seconds']/total,
                query_constraints=phase['constraints'],mean_updated_agreement=statistics.mean(x['agreement_quantized'] for x in per_set[1:]),
                mean_updated_certified_fraction=statistics.mean(x['certified_fraction'] for x in per_set[1:]) if kind=='certified' else None,
                mean_updated_interval_width=statistics.mean(x['interval_relative_width'] for x in per_set[1:]) if kind=='certified' else None,
                updated_strict_policy_accepted_sets=sum(x['coverage99_cost1_accept'] for x in per_set[1:]) if kind=='certified' else None,
                independently_replayed_samples=replay['verified'])
            summaries.append(summary)
        overhead=summaries[-1]['total_seconds']/summaries[-2]['total_seconds']-1
        summaries[-1]['certificate_total_overhead_fraction']=overhead
        summaries[-2]['certificate_total_overhead_fraction']=None

    assert summaries, 'no completed matched comparison available'
    write_csv(a.out/'matched_methods.csv',summaries);write_csv(a.out/'matched_quality_by_set.csv',setrows)

    def tex(name, caption, label, headers, rows):
        table=[r'\begin{table*}[tp]',r'\caption{'+caption+r'}\label{'+label+'}',
               r'\centering\small',r'\begin{adjustbox}{max width=\textwidth}',
               r'\begin{tabular}{@{}'+'l'*len(headers)+r'@{}}',r'\toprule',
               ' & '.join(headers)+r' \\',r'\midrule']
        table += [' & '.join(row)+r' \\' for row in rows]
        table += [r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}',r'\end{table*}']
        (a.out/name).write_text('\n'.join(table)+'\n',encoding='utf-8')

    ordered = sorted(summaries,key=lambda x:(x['dataset'],x['n'],x['rank'],x['method'],x['mode']))
    plain = {x['case']:x for x in summaries if x['mode']=='plain'}
    certified = [x for x in ordered if x['mode']=='certified']
    def catalog(x):
        return x['dataset']+r' $N='+str(x['n'])+'$'
    main = [x for x in certified if x['rank']==15 and x['basis_scale']==4096]
    tex('matched_methods.tex',
        'Completed matched audits over ten public center sets, rank 15 and $T=4096$. Each alternative pays preparation and setup once; every catalog row is proved. Component costs include native answer construction and exclude verification, fitting, warmups, I/O and public-center preparation. Full/certified compares different fidelity guarantees.',
        'tab:matched-public-full',
        ['Catalog','Method','Exact Full (s)','Plain (s)','Certified (s)','Full / certified','Cert. overhead (\\%)'],
        [[catalog(x),x['method'],f"{x['full_public_seconds']:,.1f}",f"{plain[x['case']]['total_seconds']:,.1f}",
          f"{x['total_seconds']:,.1f}",f"{x['full_over_method_ratio']:.2f}",f"{100*x['certificate_total_overhead_fraction']:.2f}"] for x in main])
    tex('matched_quality.tex',
        'Completed integer certificate quality at rank 15, $T=4096$. Agreement, certified coverage and relative cost-interval width are means across nine updated center sets. Strict accepts requires at least 99\\% certified assignments and at most 1\\% width. All checked intervals contain the exact quantized optimum; no certified assignment is incorrect.',
        'tab:matched-certificate-quality',
        ['Catalog','Method','Agreement (\\%)','Certified (\\%)','Width (\\%)','Strict accepts'],
        [[catalog(x),x['method'],f"{100*x['mean_updated_agreement']:.2f}",f"{100*x['mean_updated_certified_fraction']:.2f}",
          f"{100*x['mean_updated_interval_width']:.2f}",str(x['updated_strict_policy_accepted_sets'])+'/9'] for x in main])
    pilots = [x for x in certified if x['rank']!=15]
    if pilots:
        tex('proved_budget.tex',
            'Completed predeclared MNIST $N=256$ rank-63, $T=16384$ proof pilots (batch 16). Quality means exclude the deployed set; costs cover all ten sets. Rank, precision and batching change together, so cost differences are configuration effects rather than an isolated rank ablation.',
            'tab:proved-certificate-budget',
            ['Method','Plain (s)','Certified (s)','Full / certified','Agreement (\\%)','Certified (\\%)','Width (\\%)'],
            [[x['method'],f"{plain[x['case']]['total_seconds']:,.1f}",f"{x['total_seconds']:,.1f}",f"{x['full_over_method_ratio']:.2f}",
              f"{100*x['mean_updated_agreement']:.2f}",f"{100*x['mean_updated_certified_fraction']:.2f}",f"{100*x['mean_updated_interval_width']:.2f}"] for x in pilots])
    fullrows=sorted(full_by_root.values(),key=lambda x:('CIFAR' if x['case'].startswith('cifar') else 'MNIST',x['n']))
    write_csv(a.out/'completed_full.csv',fullrows)
    tex('completed_full.tex',
        'Completed exact public-center Full controls. One setup serves all ten center sets; every catalog batch is proved and independently replayed. Peak RSS describes the prover process tree, excluding the Python driver. Component costs include native answer construction.',
        'tab:public-full-complete',
        ['Catalog','Batch','Proofs','Constraints','Cost (s)','Peak RSS (GiB)'],
        [[('CIFAR/ResNet' if x['case'].startswith('cifar') else 'MNIST')+r' $N='+str(x['n'])+'$',str(x['batch']),str(x['proofs']),
          f"{x['constraints']:,}",f"{x['total_seconds']:,.1f}",f"{x['peak_rss_bytes']/2**30:.2f}"] for x in fullrows])
    public_replayed=sum(load(case/'public_bundle/verify.json')['verified'] for case in cases)
    scale = [x for x in main if x['dataset']=='MNIST' and x['n']==1024]
    if len(scale)==2:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
        learned=next(x for x in scale if x['method'].startswith('Learned'))
        pca=next(x for x in scale if x['method'].startswith('PCA'))
        fig,axes=plt.subplots(1,2,figsize=(10,3.5),gridspec_kw={'width_ratios':[1.25,1]})
        labels=['Exact\nFull','Learned\nplain','Learned\ncertified','PCA\nplain','PCA\ncertified']
        times=[learned['full_public_seconds'],plain[learned['case']]['total_seconds'],learned['total_seconds'],
               plain[pca['case']]['total_seconds'],pca['total_seconds']]
        bars=axes[0].bar(range(5),[x/60 for x in times],color=['#666666','#76b7b2','#167f78','#c4abd7','#79549d'])
        axes[0].set_xticks(range(5),labels);axes[0].set_ylabel('Measured component cost (min)')
        axes[0].set_ylim(0,470);axes[0].set_title('(a) Ten center sets, 1,024 MNIST records')
        axes[0].bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
        xs=np.arange(2);w=.34
        a1=axes[1].bar(xs-w/2,[100*x['mean_updated_agreement'] for x in [learned,pca]],w,label='Observed agreement',color='#4c78a8')
        a2=axes[1].bar(xs+w/2,[100*x['mean_updated_certified_fraction'] for x in [learned,pca]],w,label='Certified coverage',color='#e6a141')
        axes[1].set_xticks(xs,['Learned + energy','PCA + energy']);axes[1].set_ylabel('Assignments (%)')
        axes[1].set_ylim(0,132);axes[1].set_yticks([0,20,40,60,80,100]);axes[1].set_title('(b) Nine updated center sets')
        for bars in [a1,a2]:axes[1].bar_label(bars,fmt='%.2f',padding=3,fontsize=8)
        axes[1].legend(loc='upper center',ncol=1,fontsize=8,frameon=False)
        for ax in axes:
            ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.18)
        fig.tight_layout()
        fig.savefig(a.out/'matched_scale.pdf',bbox_inches='tight')
        fig.savefig(a.out/'matched_scale.png',dpi=180,bbox_inches='tight');plt.close(fig)
    result=dict(transfer_files_hash_checked=sum(len(x['files']) for x in captures),
                completed_exact_controls=len(full_by_root),matched_reuse_cases=len(summaries)//2,
                actual_original_integer_full_outputs_checked=True,independent_native_certificate_outputs_match=True,
                no_incorrect_certified_assignments=True,all_optimal_costs_inside_intervals=True,
                cost_policy='prepare + common Stage1 + one chosen query compile/setup + native answer + witness + median prove; no verification/warmup/I-O/fitting/public center preparation',
                source_captures=[x.relative_to(ROOT).as_posix() if x.is_relative_to(ROOT) else str(x) for x in [x.resolve() for x in capture_paths]],
                source_capture_sha256=[hashlib.sha256((x/'capture.json').read_bytes()).hexdigest() for x in capture_paths],
                total_completed_cases=len(cases),worker_public_replayed_proofs=public_replayed,
                quality_record_set_evaluations=sum(x['n'] for x in setrows),
                updated_strict_policy_accepted_sets=sum(x['coverage99_cost1_accept'] for x in setrows if x['set']!=0),
                completed_matched_methods=summaries)
    (a.out/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(out=str(a.out),complete_cases=len(cases),matched_cases=len(summaries)//2,
                          public_replayed_proofs=public_replayed,hashed_files=result['transfer_files_hash_checked'])))


if __name__=='__main__':main()
