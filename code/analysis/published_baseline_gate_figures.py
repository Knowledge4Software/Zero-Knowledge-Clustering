"""Standalone scientific summary; no new measurement or fitted extrapolation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO=Path(__file__).resolve().parents[2]
OUT=REPO/'output'
STEM='published_baseline_gate_20261029'


def heatmap(ax,values,rows,columns,title,percent=True):
    values=np.asarray(values,float)
    if percent:
        color=ax.imshow(values,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
    else:
        color=ax.imshow(values,cmap='YlOrRd',aspect='auto')
    ax.set_xticks(range(len(columns)),columns,fontsize=9)
    ax.set_yticks(range(len(rows)),rows,fontsize=8)
    ax.set_title(title,fontsize=11,loc='left',pad=9,fontweight='bold')
    for i in range(len(rows)):
        for j in range(len(columns)):
            value=values[i,j]
            label='—' if np.isnan(value) else f'{value:.2f}'
            rgba=color.cmap(color.norm(value)) if np.isfinite(value) else (1,1,1,1)
            brightness=.299*rgba[0]+.587*rgba[1]+.114*rgba[2]
            ax.text(j,i,label,ha='center',va='center',fontsize=9,
                    color='white' if brightness<.5 else '#182332')
    ax.tick_params(length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    return color


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stem',default=STEM)
    args=parser.parse_args();stem=args.stem
    report=json.loads((OUT/(stem+'.json')).read_text())
    assert report['native_complete']
    native={(r['dataset'],r['method']):r for r in report['native_summary'] if r['n']==128}
    datasets=['fashion','text','har'];labels=['Fashion','Text','HAR']
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    methods=['ordered_head_certificate','pca_certificate','sign_rp_raw','sparse_rp_raw','rabitq_raw']
    names=['Current centroid','PCA','Sign projection','Sparse projection','RaBitQ (one bit)']
    values=[[100*native[(d,m)]['assignment_agreement'] if (d,m) in native else np.nan for d in datasets] for m in methods]
    heatmap(axes[0,0],values,names,labels,'A  Agreement with exact labels (%)')
    methods+=['lightweight_coreset_m16','lightweight_coreset_m32','uniform_coreset_m32']
    names+=['Lightweight coreset,16','Lightweight coreset,32','Uniform sample,32']
    values=[[100*native[(d,m)]['balanced_sign_agreement'] if (d,m) in native else np.nan for d in datasets] for m in methods]
    heatmap(axes[0,1],values,names,labels,'B  Balanced estimated-sign agreement (%)')
    methods=['ordered_head_certificate','pca_certificate','sign_rp_certificate','sparse_rp_certificate']
    names=['Current centroid','PCA','Sign RP + our wrapper','Sparse RP + our wrapper']
    values=[[100*native[(d,m)]['refined_coverage'] for d in datasets] for m in methods]
    heatmap(axes[1,0],values,names,labels,'C  Deterministic coverage after ≤16 corrections (%)')
    methods=['Exact Full / Groth16','Current centroid certificate','PCA certificate',
             'Sign RP + our certificate adapter','Sparse RP + our certificate adapter','ZKML primitive adapter / Halo2']
    names=['Full / Groth16','Current centroid','PCA','Sign RP + our wrapper','Sparse RP + our wrapper','ZKML adapter / Halo2']
    proof={(r['dataset'],r['method']):r for r in report['proof_rows'] if r['complete']}
    values=[[proof[(d,m)]['seconds']/60 if (d,m) in proof else np.nan for d in datasets[:2]] for m in methods]
    heatmap(axes[1,1],values,names,labels[:2],'D  Measured cold proof-service total (minutes)',percent=False)
    fig.suptitle('Local gate against paper-derived baselines',fontsize=16,fontweight='bold')
    fig.supxlabel('A–C:128 records,8 requests ×3 exploratory fits. D:32 records,4 requests,one timing execution.\n'
        'Point estimates are not certificates. RaBitQ statistical intervals are reported separately.\n'
        'Raw algorithms and proof adapters have different output contracts.',fontsize=9)
    for extension in ['png','svg']:
        fig.savefig(OUT/(stem+'_summary.'+extension),dpi=160)
    plt.close(fig)
    print('Saved standalone PNG and SVG summary.')


if __name__=='__main__':main()
