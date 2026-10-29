"""Presentation-only revision of an already checked, complete result export.

No fitting, native certificate recomputation, proof generation or proof replay.
Existing numerical table cells are retained; known-bank affine is separated.
"""
from pathlib import Path
import json,re
from campaigns.local_fast_tradeoff import save,sha
from analysis.export_main_rank79_paper import table


def reframe(repo,dest,campaign):
    receipt=json.loads((dest/'export_receipt.json').read_text())
    assert receipt['all_matched_complete'] and len(receipt['complete_matched_cohorts'])==3
    for name,digest in receipt['sources'].items():assert sha(repo/name)==digest,'Changed evidence: '+name
    captions={
        'completed_proofs':'Completed real-proof controls, $N=128$, two Go workers and one repetition per workload. Active minutes include cold preparation, setup and fresh public replay. Neither cohort triggers fallback. Future candidate centers are not required by preprocessing.',
        'matched_results':'Matched unknown-future-center controls against one fixed reference, eight Go workers and four distinct proposals per cohort. Centroid and PCA are two bases for the same rank-79 certificate with rank-63 prediction; Full is exact public-center evaluation. Mature and early mean reference Lloyd ages 8 and 1; scale is the larger age-1 catalog. Concl. counts decisions before exact fallback (FB); proof counts include paid FB. Min. denotes active minutes. Diagnostic fits were observed during development.',
        'matched_quality':'Quality in matched unknown-future-center controls. Percentage columns concern different quantities: agreement, assignment-certified fraction, native cost error and proved change-interval width. Width is a percentage of old exact cost, before fallback. Moments-only has no label predictor or comparable two-sided width. A dash in the regression column means no regression was present.',
        'matched_components':'Phase costs in minutes, including all recorded attempts. Prep includes fitting where used, commitments, cold key work and original-row proofs; Full compile and setup time is separated from query. Query retains native parameters and key I/O. Replay is fresh public-only checking; FB is newly generated exact fallback. Centroid and PCA denote bases of the same certificate; mature/early are reference Lloyd ages 8/1.'}
    previous={}
    for name,caption in captions.items():
        path=dest/(name+'.tex');previous[path.name]=sha(path);lines=[]
        for line in path.read_text().splitlines():
            if ('Family affine' in line or 'Known-family affine' in line) and '&' in line:continue
            if line.startswith('\\caption{'):line='\\caption{'+caption+'}'
            if name=='matched_quality' and '& Moments-only &' in line:
                cells=line.split('&');assert len(cells)==8;cells[6]=' -- ';line='&'.join(cells)
            if name=='matched_quality':line=re.sub(r'(?<!\d)0/0','--',line)
            lines.append(line)
        path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    def read(p):return json.loads(p.read_text())
    old=read(repo/'code/validation/outputs/known_family_affine_n128_20261024/checked_comparison.json')
    rows=[f"MNIST,19 & 128 & 2 & 85 & {old['query_center_sets']} & {old['family']['active_phase_wall_seconds']/60:.3f} & {old['family']['public_proofs']}"]
    ratios=[];extension=read(campaign/'refinement_extension.json')
    for name,label,n in [('mnist_switching_mature_n128','MNIST mature',128),('fashion_early_n128','Fashion early',128),('mnist_early_n1024','MNIST scale',1024)]:
        root=campaign/name/'known_family_affine';r=read(root/'checked_comparison.json');assert r['complete'] and r['all_outputs_match_full']
        rows.append(f"{label} & {n} & 8 & {read(root/'protocol.json')['bank_size']} & {r['query_center_sets']} & {r['family']['active_phase_wall_seconds']/60:.3f} & {r['family']['public_proofs']}")
        q=read(Path(extension['root'])/name/'head79/checked_comparison.json');assert q['complete'];ratios.append(q['full_over_group_active_phase'])
    (dest/'known_family.tex').write_text(table('Supplementary exact affine measurements with advance knowledge of every future center. The bank includes all evaluated center sets before preprocessing; sets include the deployed reference. These are a different information contract from the main unknown-future-center task. No online-affine speed comparison is inferred. Assignments and costs are exact within the bank.','tab:known-family','lrrrrrr',r'Cohort & $N$ & Workers & Bank & Sets & Min. & Proofs',rows,True),encoding='utf-8')
    (dest/'headline_values.tex').write_text('\\newcommand{\\MatchedMinRatio}{'+f'{min(ratios):.2f}'+'}\n\\newcommand{\\MatchedMaxRatio}{'+f'{max(ratios):.2f}'+'}\n',encoding='utf-8')
    for path in dest.glob('*.tex'):
        text=path.read_text()
        if path.name=='matched_components.tex' and r'\setlength{\tabcolsep}{8pt}' not in text:
            text=text.replace('\\centering\n','\\centering\n'+r'\setlength{\tabcolsep}{8pt}'+'\n',1)
        for old,new in [('Group79','Centroid'),('PCA79','PCA'),('Group 79','Centroid'),(' & Head &',' & Centroid &'),('(head)','(centroid)'),('Head ratio range','Centroid ratio range'),('Head min.','Centroid min.'),('Full/Head','Full/Centroid')]:text=text.replace(old,new)
        if path.name=='completed_quality.tex':text=text.replace('Selected group-79 controls:','Centroid-basis certificate controls (rank 79; prediction rank 63):')
        if path.name in ['native_quality.tex','confirmation_quality.tex']:
            text=text.replace('\\caption{','\\caption{Centroid and PCA denote the two bases of the same certificate. ',1)
            # Keep repeated presentation-only builds idempotent.
            text=text.replace('Centroid and PCA denote the two bases of the same certificate. Centroid and PCA denote the two bases of the same certificate. ','Centroid and PCA denote the two bases of the same certificate. ')
        if path.name=='refinement_pilots.tex':text=text.replace('Completed selected-method/Full pairs','Completed basis-specific certificate/Full pairs').replace('Selected-method cost','Certificate cost').replace('same selected-method/Full pairs','same basis-specific certificate/Full pairs')
        path.write_text(text,encoding='utf-8')
    receipt['presentation_amendment']=dict(date='2026-10-26',known_family_separated=True,representation_agnostic=True,online_affine_measured=False,measurements_recomputed=False,previous_table_sha256=previous)
    replay=campaign/'public_replay_round3_20261026/status.json'
    if replay.exists():
        check=read(replay)
        assert check['complete'] and check['verified_total']==sum(r['verified'] for r in check['results'])
        receipt['additional_replay']=dict(source=replay.relative_to(repo).as_posix(),sha256=sha(replay),verified=check['verified_total'],verification_only=True)
    save(dest/'export_receipt.json',receipt)
    return receipt
