"""Build a transparent text failure report; validation improvements stay separate."""
import argparse
import json
from pathlib import Path
import shutil

import pandas as pd

from commitjl.reusable_native import file_hash
from experiments.laptop_comparison import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--diagnosis',type=Path,required=True);args=parser.parse_args();root=args.root
    d=json.loads(args.diagnosis.read_text());manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['complete'] and manifest['cases']==720 and manifest['test_evaluations']==0
    code=Path(__file__).resolve().parents[1]
    for name,digest in manifest['identity']['sources'].items():
        assert file_hash(code/name)==digest,name
        target=root/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(code/name,target)
    summary=pd.read_csv(root/'summary.csv');norm=pd.read_csv(root/'norm_corrected_summary.csv')
    raw=pd.read_csv(root/'rows.csv');assert not raw.any_iteration_cap.any()
    selections=[('Current +12','protected',12,1,1.),('Denser JL, q=4','protected',12,4,1.),
        ('64 extra directions','protected',64,1,1.),('Tail weight 0.5','protected',12,1,.5),
        ('Tail weight 0.25','protected',12,1,.25),('Fast PCA','pca_fast',0,0,1.),('Raw LSA','lsa',0,0,1.)]
    points=[]
    for m in [128,256]:
        for budget in [1,5]:
            for label,method,extra,q,weight in selections:
                r=summary[(summary.m==m)&(summary.init_budget==budget)&(summary.method==method)&
                    (summary.extra==extra)&(summary.q==q)&(summary.weight==weight)].iloc[0]
                points.append(dict(m=m,init_budget=budget,variant=label,ari=r.validation_ari,
                    ari_low=r.validation_ari_ci_low,ari_high=r.validation_ari_ci_high,
                    pair_q95=r.pair_q95,ground_truth_ari=r.ground_truth_ari))
            r=norm[(norm.m==m)&(norm.init_budget==budget)&(norm.q==1)].iloc[0]
            points.append(dict(m=m,init_budget=budget,variant='Residual norm correction',ari=r.validation_ari,
                ari_low=r.validation_ari_ci_low,ari_high=r.validation_ari_ci_high,pair_q95=r.pair_q95,ground_truth_ari=None))
    frame=pd.DataFrame(points);frame.to_csv(root/'diagnostic_comparison.csv',index=False)
    headers=['m','Initializations','Variant','Validation ARI','95% interval','Pair-distance Q95 error']
    table=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    for r in points:
        table.append(f'| {r["m"]} | {r["init_budget"]} | {r["variant"]} | {r["ari"]:.4f} | '+
            f'[{r["ari_low"]:.4f}, {r["ari_high"]:.4f}] | {r["pair_q95"]:.4f} |')
    energy=[a['centered_training_energy_fraction'] for a in d['protected_training_energy'] if a['kind']=='enriched']
    lines=['# Why text clustering is difficult for the current method','',
        '## Measured diagnosis','',
        f'The nonzero archived corpus has {d["n"]} documents and {d["d"]} TF-IDF features. '+
        f'The typical document has {d["nonzero_features_per_document"]["median"]:.0f} nonzero features. '+
        'Every row already has unit L2 norm; missing input normalization is not the cause. '+
        f'Median sampled cosine similarity is {d["sampled_cosine_similarity"]["median"]:.4f}, '+
        'and median sampled squared Euclidean distance is approximately 2. Many document pairs are tied or nearly tied.', '',
        f'The protected rank-16 head captures only {100*min(energy):.2f}--{100*max(energy):.2f}% of centered training variation. '+
        'The remaining approximately 93.4% is represented through the random residual sketch. '+
        'Inference: preserving overall squared distances can still perturb weak cluster separations and change fresh KMeans labels. '+
        'Preserving assignments to the original five fixed centers does not preserve the path of fresh KMeans, whose centers move.', '',
        f'Full itself has median between-start ARI {d["full_pairwise_ari"]["median"]:.3f} '+
        f'(range {d["full_pairwise_ari"]["min"]:.3f}--{d["full_pairwise_ari"]["max"]:.3f}). '+
        'Thus Full self-agreement=1 in the main table is a reference definition, not proof of a unique or semantically correct clustering. '+
        'The corpus has 20 source classes while the current primary comparison requests five clusters. '+
        'This does not invalidate equal-k method comparisons, but source-label ARI must not be confused with agreement with Full.', '',
        'At m=256 in the completed whole-corpus replication, +12 had ARI 0.0483 versus fast PCA 0.3695. '+
        'However, its original-space partition SSE was only 1.0080 times Full, and pair squared-distance Q95 error was 0.1708 versus PCA 0.8437. '+
        'This is poor partition agreement, not a failure of every compression metric. The objective and labels tell different stories.', '',
        '## Candidate fixes tested','',
        '720 validation-only cases: m=128/256, extra directions=12/32/64, q=1/4, residual weights=1/0.5/0.25, '+
        'three new projection draws crossed with three new training/clustering starts, and equal one/five-initialization budgets. '+
        'Fast dedicated-rank PCA and unnormalized LSA are controls. All models are learned on training rows. '+
        'No test-fold outcome selects or promotes a candidate. The fixed datasets were explored before; this is not independent-data confirmation.', '',
        'A separate 72-case paired follow-up restores the norm of each original orthogonal residual after JL. '+
        'It leaves the protected head intact and preserves original fixed-center squared distances in real arithmetic. '+
        'The implementation rejects an annihilated positive residual. It is nonlinear postprocessing, requires original norms, '+
        'and is not established by the original linear-JL/ZK argument.', '',
        '\n'.join(table), '',
        '## What the tests support','',
        'Denser JL, additional directions, and norm correction do not provide a general solution for fresh text clustering in this screen. '+
        'Residual norm correction improves distance fidelity but leaves ARI weak. Downweighting the tail can improve validation ARI, '+
        'including against some PCA controls, while causing much larger distance error. '+
        'Consequently, a higher ARI for that candidate cannot be presented as successful Euclidean distance preservation. '+
        'Single-observation prototype timings are diagnostic and cannot establish a speed victory.', '',
        'The next defensible steps are to evaluate stable multi-start, text-appropriate angular/LSA clustering equally for all methods, '+
        'and to explore task-aware protection of additional prototype/decision directions. '+
        'Those change the clustering protocol or learned protected subspace and require separate validation and full cost accounting. '+
        'The current method is retained unchanged in the complete native campaign. No clean fix is claimed here.', '',
        'Scikit-learn documents high-dimensional text KMeans instability and the use of LSA followed by normalization: '+
        '[official text clustering example](https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html). '+
        'An angular/normalized experiment would be a separate task, with the same treatment for Full, PCA and our method.', '',
        'Raw geometry diagnosis: `../text_geometry_diagnosis_20261016/diagnosis.json`. '+
        'Complete grid: `rows.csv`, `summary.csv`. Norm correction: `norm_corrected_rows.csv`, `norm_corrected_summary.csv`. '+
        'Validation-fold and whole-corpus ARIs have different scopes and must not be compared as if they measured the same task.','']
    (root/'findings.md').write_text('\n'.join(lines),encoding='utf-8')
    artifacts={str(p.relative_to(root)):file_hash(p) for p in (root/'artifacts').glob('*.npz')}
    atomic_json(root/'report_manifest.json',dict(complete=True,validation_cases=720,norm_correction_cases=72,
        test_evaluations=0,artifact_hashes=artifacts,
        outputs={name:file_hash(root/name) for name in ['findings.md','diagnostic_comparison.csv','summary.csv',
            'rows.csv','norm_corrected_summary.csv','norm_corrected_rows.csv']}))
    print(json.dumps(dict(report=str(root/'findings.md'),artifact_hashes=len(artifacts))))


if __name__=='__main__':main()
