"""Automatic descriptive tables from verified final-case records only."""
from pathlib import Path
import csv
import io
import statistics
import time

from experiments.pca_advantage_20261102 import read,save,sha

ROOT=Path(__file__).resolve().parents[2]/'code/validation/outputs/server_final_method_20261102'
METHODS=['ordered_head','pca','full','adapter']


def write_csv(path,records):
    if not records:
        path.write_text('');return
    text=io.StringIO();writer=csv.DictWriter(text,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    path.write_text(text.getvalue())


def update():
    records=[];pins={}
    for p in (ROOT/'cases').glob('*/result.json'):
        r=read(p)
        if r.get('complete') and r.get('all_decisions_correct'):
            records.append(r);pins[str(p.relative_to(ROOT))]=sha(p)
    records.sort(key=lambda r:(r['dataset'],r['seed']))
    report=ROOT/'reports';report.mkdir(exist_ok=True)
    per_fit=[];crossovers=[]
    for r in records:
        row=dict(dataset=r['dataset'],seed=r['seed'],requests=8,our_rank=r['ranks']['ordered_head'],pca_rank=r['ranks']['pca'])
        for m in METHODS:
            row[m+'_warm_seconds']=r[m]['warm_seconds'];row[m+'_cold_component_seconds']=r[m]['cold_component_seconds']
        row['ours_certificate_requests']=r['ordered_head']['certificate_requests']
        row['pca_certificate_requests']=r['pca']['certificate_requests'];per_fit.append(row)
        for baseline in ['pca','full','adapter']:
            a,b=r['ordered_head'],r[baseline]
            setup_a=a['cold_component_seconds']-a['warm_seconds'];setup_b=b['cold_component_seconds']-b['warm_seconds']
            saving=b['warm_seconds']-a['warm_seconds']
            # Integer batches of eight, under a stationary future request mixture.
            batches=max(1,int((setup_a-setup_b)//saving)+1) if saving>0 else None
            crossovers.append(dict(dataset=r['dataset'],seed=r['seed'],baseline=baseline,
                modeled_batches_of_eight=batches,modeled_requests=(8*batches if batches else None),
                steady_state_warm_saving_per_batch=saving,measured_threshold=False))
    aggregate=[]
    for dataset in ['fashion','har','mnist','text']:
        fits=[r for r in records if r['dataset']==dataset]
        if not fits:continue
        row=dict(dataset=dataset,completed_fits=len(fits),planned_fits=3)
        for m in METHODS:
            v=[r[m]['warm_seconds'] for r in fits]
            row[m+'_mean_warm_seconds']=statistics.mean(v)
            row[m+'_min_warm_seconds']=min(v);row[m+'_max_warm_seconds']=max(v)
        for b in ['pca','full','adapter']:row[b+'_over_ours_mean_time_ratio']=row[b+'_mean_warm_seconds']/row['ordered_head_mean_warm_seconds']
        aggregate.append(row)
    write_csv(report/'per_fit.csv',per_fit);write_csv(report/'per_dataset.csv',aggregate)
    write_csv(report/'modeled_crossover.csv',crossovers)
    lines=['Verified final-method results','',f'{len(records)} of 12 planned fitted configurations completed.',
        '', 'Warm seconds cover eight candidate updates with cached setup. Cold component sums '
        'include the measured fitting/validation, branch authentication and any required Full initialization. '
        'One warm repetition per fit; ranges across three fits are descriptive.','',
        '| Dataset | Completed fits | Ours mean (s) | PCA mean (s) | Full mean (s) | Adapter mean (s) |',
        '|---|---:|---:|---:|---:|---:|']
    for r in aggregate:lines.append(f"| {r['dataset']} | {r['completed_fits']}/3 | {r['ordered_head_mean_warm_seconds']:.1f} | {r['pca_mean_warm_seconds']:.1f} | {r['full_mean_warm_seconds']:.1f} | {r['adapter_mean_warm_seconds']:.1f} |")
    lines+=['','The ZKML primitive adapter uses a different commitment and stronger exact-cost output. '
       'Fixed-rank numerical certificate coverage, validation-selected service time and classification '
       'accuracy are distinct outcomes. All failed or excluded cases remain in the campaign records.',
       '', 'Crossover values are projections for repeated batches of eight under an unchanged '
       'catalog/reference and stationary request mixture; they are not measured large-request experiments. '
       'Changing the reference requires charging new setup. No universal speed superiority is inferred.']
    (report/'final_results.md').write_text('\n'.join(lines)+'\n')
    save(report/'analysis_pin.json',dict(updated_unix=time.time(),result_sha256=pins,
         analysis_source_sha256=sha(Path(__file__)),completed_fits=len(records)))
    return len(records)


def main():
    while not(ROOT/'plan.json').exists():time.sleep(5)
    deadline=read(ROOT/'plan.json')['rental_expiry_estimate_unix']
    while time.time()<deadline:
        update()
        if(ROOT/'finished.json').exists():break
        time.sleep(30)


if __name__=='__main__':main()
