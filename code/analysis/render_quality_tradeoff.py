"""Render quality measurements without an application-specific acceptance rule.

The original numerical CSV and its historical policy columns are preserved.
"""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = ROOT/'paper_results/review_controls/generated_completed_20261020/matched_methods.csv'
    frame = pd.read_csv(source)
    frame = frame[(frame['mode']=='certified') & (frame['rank']==15)]
    if len(frame)!=8 or set(frame['basis_scale'])!={4096}:
        raise ValueError('unexpected completed main comparison')
    rows = []
    for _,row in frame.iterrows():
        if row['dataset'] not in ('MNIST','CIFAR/ResNet') or row['method'] not in ('Learned + energy','PCA + energy'):
            raise ValueError('unexpected table label')
        rows.append(f"{row['dataset']} $N={int(row['n'])}$ & {row['method']} & "
                    f"{100*row['mean_updated_agreement']:.2f} & "
                    f"{100*row['mean_updated_certified_fraction']:.2f} & "
                    f"{100*row['mean_updated_interval_width']:.2f}" + r' \\')
    text = r'''\begin{table*}[tp]
\caption{Completed integer certificate quality at rank 15, $T=4096$. Agreement, certified coverage and relative cost-interval width are means across nine updated center sets. All checked intervals contain the exact quantized optimum; no certified assignment is incorrect.}\label{tab:matched-certificate-quality}
\centering\small
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{@{}llrrr@{}}
\toprule
Catalog & Method & Agreement (\%) & Certified (\%) & Width (\%) \\
\midrule
''' + '\n'.join(rows) + '\n' + r'''\bottomrule
\end{tabular}
\end{adjustbox}
\end{table*}
'''
    out = ROOT/'paper_results/review_controls/presentation_20261021'
    out.mkdir(parents=True,exist_ok=True)
    target = out/'matched_quality.tex'
    target.write_bytes(text.encode('utf-8'))
    (out/'manifest.json').write_text(json.dumps(dict(
        source=source.relative_to(ROOT).as_posix(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        output=target.name,output_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),rows=len(rows),
        change='Presentation only: omit illustrative acceptance rule; preserve all agreement, coverage and cost-width measurements. Original policy fields remain in frozen evidence.'),indent=2),encoding='utf-8')
    print(target)


if __name__=='__main__':
    main()
