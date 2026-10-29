"""Editable paper SVGs and evidence-backed pilot plots; no manuscript edits.

All diagrams describe the current group certificate, not historical sparse JL.
Diagram values are illustrative. Plots read measured receipts and are explicitly
labeled exploratory/native-only as applicable. No server jobs are launched.
"""
from pathlib import Path
import csv
import hashlib
import json
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

REPO=Path(__file__).resolve().parents[2]
OUT=REPO/'output/paper_figures_20261029'
NAVY='#18354D'; BLUE='#236A95'; TEAL='#217F73'; AMBER='#AD6B1B'
INK='#263442'; GRAY='#677888'; PALE='#EAF3F9'; GREEN='#E9F5F0'; SAND='#FFF3DF'
SOURCES={}
FIGURES=[]


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(relative):
    p=REPO/relative;SOURCES[relative]=sha(p)
    return json.loads(p.read_text(encoding='utf-8'))


def theme():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.6,'text.color':INK,
        'axes.labelcolor':INK,'axes.edgecolor':GRAY,'xtick.color':INK,'ytick.color':INK,
        'svg.fonttype':'none','svg.hashsalt':'prototype-update-figures-20261029',
        'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})


def canvas(title,subtitle,height=3.9):
    fig=plt.figure(figsize=(7.4,height),facecolor='white')
    ax=fig.add_axes([.025,.085,.95,.77]);ax.set(xlim=(0,100),ylim=(0,100));ax.axis('off')
    fig.text(.025,.953,title,color=NAVY,fontsize=11.4,weight='bold',va='top')
    fig.text(.025,.889,subtitle,color=GRAY,fontsize=7.1,va='top')
    return fig,ax


def card(ax,x,y,w,h,title,body='',color=BLUE,fill=PALE,heading=7.9,body_size=7.0):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.25,rounding_size=1.1',
                              facecolor=fill,edgecolor=color,linewidth=.85))
    title_artist=ax.text(x+1.8,y+h-3,title,ha='left',va='top',fontsize=heading,color=color,weight='bold')
    checks=getattr(ax.figure,'_card_checks',[])
    checks.append((ax,title_artist,(x+1,y+1,x+w-1,y+h-1)))
    if body:
        artist=ax.text(x+1.8,y+h-10,body,ha='left',va='top',fontsize=body_size,linespacing=1.5)
        checks.append((ax,artist,(x+1,y+1,x+w-1,y+h-1)))
    ax.figure._card_checks=checks
    return (x,y,w,h)


def label(ax,x,y,text,color=GRAY,size=7.0,**kwargs):
    return ax.text(x,y,text,color=color,fontsize=size,**kwargs)


def arrow(ax,start,end,color=GRAY,dashed=False,rad=0):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=8,
        linewidth=.9,color=color,linestyle='--' if dashed else '-',
        connectionstyle=f'arc3,rad={rad}',shrinkA=1.5,shrinkB=1.5))


def footer(fig,text):
    fig.text(.025,.025,text,fontsize=6.7,color=GRAY,va='bottom')


def export(fig,name,caption,kind):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();bounds=fig.bbox
    # Fit text to its own card as well as the page, leaving editable text.
    for ax,artist,box in getattr(fig,'_card_checks',[]):
        for _ in range(15):
            extent=artist.get_window_extent(renderer)
            x0,y0=ax.transData.transform(box[:2]);x1,y1=ax.transData.transform(box[2:])
            if extent.x0>=x0-1 and extent.y0>=y0-1 and extent.x1<=x1+1 and extent.y1<=y1+1:break
            artist.set_fontsize(artist.get_fontsize()-.15)
            if artist.get_fontsize()<6.2:raise RuntimeError(f'{name}: shorten card text: '+artist.get_text())
        else:raise RuntimeError(name+': card text could not be fitted')
    fig.canvas.draw()
    clipped=[]
    for text in fig.findobj(match=matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            extent=text.get_window_extent(renderer)
            if extent.x0<bounds.x0-2 or extent.y0<bounds.y0-2 or extent.x1>bounds.x1+2 or extent.y1>bounds.y1+2:
                clipped.append(text.get_text())
    if clipped:raise RuntimeError(f'{name}: labels outside canvas: {clipped}')
    svg=OUT/(name+'.svg');png=OUT/(name+'.png')
    fig.savefig(svg,metadata={'Date':None,'Description':caption})
    fig.savefig(png,dpi=200)
    document=ET.parse(svg)
    if any(node.tag.endswith('}image') for node in document.iter()):
        raise RuntimeError('SVG contains embedded raster content: '+name)
    FIGURES.append(dict(name=name,kind=kind,caption=caption,svg=svg.name,png=png.name,
         svg_sha256=sha(svg),png_sha256=sha(png),labels_within_canvas=True,embedded_raster=False))
    plt.close(fig)


def intro():
    fig,ax=canvas('Repeated prototype audits: where reuse can help',
                 'The catalog and deployed reference stay fixed; each request supplies a new candidate bank.')
    label(ax,1,96,'DIRECT FULL AUDIT',color=GRAY,size=8,weight='bold')
    label(ax,52,96,'REUSABLE CERTIFICATE',color=TEAL,size=8,weight='bold')
    for y,q in [(66,'1'),(39,'2'),(12,'Q')]:
        card(ax,1,y,11,21,'$C^{'+q+'}$','',color=GRAY,fill='#F5F7F9',heading=9)
        card(ax,17,y,29,21,'Prove exact full-space cost','Original rows, every request',color=GRAY,fill='#F5F7F9')
        arrow(ax,(12,y+10),(17,y+10))
    card(ax,52,61,46,26,'Authenticate reusable state once',
         'Private heads + residual energies + group moments\nExact old membership bound to the original catalog',color=TEAL,fill=GREEN)
    for x,t in [(52,'$C^1$'),(68,'$C^2$'),(84,'$C^Q$')]:
        card(ax,x,24,14,23,t+' audit','Compact proof\nor direct Full',color=BLUE,fill=PALE,body_size=6.8)
        arrow(ax,(x+7,61),(x+7,47),color=TEAL)
    label(ax,52,11,'Reuse pays only if savings cover preparation and fallback.',size=7.0,color=TEAL)
    ax.plot([49,49],[7,97],color='#C7D2DC',lw=.8)
    footer(fig,'One cache epoch: fixed X, reference C⁰, basis and quantization. Changing that contract requires new authenticated state.')
    export(fig,'01_intro_reuse','Conceptual motivation for repeated audits. New candidate banks are variable; reuse is conditional on a fixed authenticated epoch. No runtime or universal speed advantage is implied.','schematic')


def pipeline():
    fig,ax=canvas('Method overview: authenticate once, verify changing candidates',
                 'The high-dimensional assigned contribution is exact; uncertainty is isolated to switching.',height=4.45)
    label(ax,1,96,'PREPARATION FOR ONE FIXED EPOCH',color=TEAL,size=8,weight='bold')
    card(ax,1,56,30,33,'1  Fit and freeze representation',
         'Training data + deployed reference C⁰\nPrediction basis B₆₃; certificate B₇₉\nFix quantization and integer scale T',color=TEAL,fill=GREEN)
    card(ax,35,56,30,33,'2  Build private summaries',
         'Catalog X → heads H and energies E\nExact old nearest groups g\nGroup counts n and sums S',color=TEAL,fill=GREEN)
    card(ax,69,56,30,33,'3  Commit and authenticate',
         'Salted original / derived / moment roots\nAuditor challenge ξ after commitment\nProve every row and its statistics',color=TEAL,fill=GREEN,body_size=6.8)
    arrow(ax,(31,72),(35,72),TEAL);arrow(ax,(65,72),(69,72),TEAL)
    label(ax,1,43,'EACH APPROVED CANDIDATE BANK Cᵗ',color=BLUE,size=8,weight='bold')
    card(ax,1,5,30,31,'4  Exact assigned change A',
         'Private counts and coordinate sums\nApproved original-space centers\nProve moment linkage + exact change')
    card(ax,35,5,30,31,'5  Bound switching corrections',
         'Rounded-basis Gram correction\nInteger residual-radius envelopes\nL ≤ T⁶Δ ≤ U, with Δ = A + Σ bᵢ')
    card(ax,69,5,30,31,'6  Verify the decision',
         'Check roots, keys and full row coverage\nConclusive? Refine ≤16 exact rows\nor prove an exact Full fallback',color=AMBER,fill=SAND,body_size=6.8)
    arrow(ax,(31,20),(35,20));arrow(ax,(65,20),(69,20))
    arrow(ax,(84,56),(84,49),TEAL)
    label(ax,83,48,'Reusable authenticated state',ha='right',va='top',size=6.7,color=TEAL)
    footer(fig,'Current centroid / PCA certificate profile. Candidate values may leave the fitted span; larger residuals can make bounds inconclusive.')
    export(fig,'02_methodology_pipeline','The current six-stage group cost-change certificate. The auditor challenge follows commitment. Complete row coverage binds exact old membership and private summaries. Prediction rank63 and certificate rank79 describe the current profile, not a universal rank choice.','schematic')


def refinement():
    fig,ax=canvas('Why bounded refinement can resolve an update',
                 'Exact-row corrections shrink an authenticated interval without changing its guarantee.',height=3.65)
    card(ax,1,16,40,72,'Exact decomposition',
         'Δ = A + Σ bᵢ\n\nA: exact assigned-center change\nbᵢ ≤ 0: benefit from switching centers\n\nRefine rows by descending interval width.\nEvery exact correction needs its own proof.',body_size=7.4)
    x=lambda z:74+z*6
    ax.plot([x(-4),x(4)],[19,19],color=GRAY,lw=.8)
    for v in [-4,-2,0,2,4]:
        ax.plot([x(v),x(v)],[18,20],color=GRAY,lw=.8);label(ax,x(v),12,str(v),ha='center')
    ax.plot([x(0),x(0)],[22,89],color=GRAY,lw=.8,ls='--')
    label(ax,73,93,'Intervals shown in cost units',ha='center',color=GRAY,size=7)
    for lo,hi,y,color in [(-3,3,69,BLUE),(-1.8,-.4,39,TEAL)]:
        ax.plot([x(lo),x(hi)],[y,y],lw=6,color=color,solid_capstyle='butt',alpha=.20)
        ax.plot([x(lo),x(hi)],[y,y],lw=1.5,color=color)
        ax.scatter([x(lo),x(hi)],[y,y],s=16,color=color,zorder=3)
    label(ax,50,80,'Before: contains zero',color=BLUE,size=7.5)
    label(ax,50,50,'After: certifies improvement',color=TEAL,size=7.5)
    label(ax,74,3,'Negative Δ = lower catalog cost',ha='center',size=7)
    footer(fig,'Illustrative intervals, not experimental results. If the budget does not resolve the sign, the service must prove an exact fallback.')
    export(fig,'03_bounds_refinement','Illustration of interval refinement after dividing integer endpoints by T⁶. A base interval spanning zero is inconclusive; authenticated exact-row corrections can establish a negative upper endpoint. The drawn endpoints are illustrative.','schematic')


def epochs():
    fig,ax=canvas('Measure amortization within the lifetime of a valid cache',
                 'Count native compression, authentication, key work, query work and every fallback.',height=3.7)
    card(ax,1,62,30,27,'K: circuit / key setup','Compile, generate and serialize keys\nReuse only under valid model / key pins',color=GRAY,fill='#F3F5F7')
    card(ax,35,62,30,27,'P: epoch preparation','Fit, summarize, commit, authenticate\nRows, old groups and private moments',color=TEAL,fill=GREEN)
    card(ax,69,62,30,27,'R: changing-request work','Plan + witness + prove + verify + I/O\nCharge selected corrections and Full')
    label(ax,50,48,'T(Q) = K + P + L(Q) + Σₜ Rₜ',ha='center',size=12,color=NAVY,weight='bold')
    label(ax,50,38,'L(Q) includes process / batch key loading; it is not key generation.',ha='center',size=7)
    for x,t,fill,color in [(1,'Epoch 1: X, C⁰, B',GREEN,TEAL),(35,'Epoch 2: changed reference',SAND,AMBER),(69,'Epoch 3: changed catalog',SAND,AMBER)]:
        card(ax,x,1,30,25,t,'Prepare again when the cache changes\nNew setup too if circuit/model pins change',color=color,fill=fill,body_size=6.6)
    arrow(ax,(31,14),(35,14));arrow(ax,(65,14),(69,14))
    footer(fig,'A long sequence amortizes preparation only while its authenticated contract stays valid. More requests do not guarantee a crossing.')
    export(fig,'04_epoch_cost_accounting','Cost boundaries and cache epochs. Changing a reference, basis, coordinate convention or catalog invalidates affected authenticated state; key setup must also be charged when model/circuit pins no longer permit reuse. L denotes recurring loading overhead, not setup.','schematic')


def routing():
    fig,ax=canvas('Engineering variant: choose proof work before proving',
                 'The planner is untrusted. Both branches must verify against the same catalog and approved requests.',height=3.9)
    card(ax,1,49,29,38,'Native preflight',
         'Cached heads / energies / moments\nPlan ≤16 exact-row corrections\nEstimate proofs + key loading\nCharge this native computation',body_size=7)
    ax.add_patch(Polygon([[49,90],[63,69],[49,48],[35,69]],closed=True,facecolor=SAND,edgecolor=AMBER,lw=.9))
    label(ax,49,69,'Conclusive and\ncheaper batch?',ha='center',va='center',color=AMBER,size=8,weight='bold')
    arrow(ax,(30,69),(35,69))
    card(ax,70,65,29,25,'Compact + exact corrections','Generate and verify actual proofs',color=TEAL,fill=GREEN,body_size=6.8)
    card(ax,70,28,29,25,'Direct exact Full','Skip failed compact work',color=GRAY,fill='#F3F5F7',body_size=6.8)
    arrow(ax,(61,73),(70,77),TEAL);label(ax,64,81,'yes',color=TEAL,size=6.8)
    arrow(ax,(56,58),(70,40),GRAY);label(ax,62,48,'no',size=6.8)
    card(ax,1,1,58,28,'Public verifier: derive the answer from proofs',
         'Same raw root and approved centers; complete disjoint request coverage\nNo answer is accepted solely because the native planner predicted it.',color=NAVY,fill=PALE,body_size=6.9)
    ax.plot([84,84],[28,18],color=GRAY,lw=.9)
    ax.plot([99,99.7,99.7,84],[77,77,18,18],color=TEAL,lw=.9)
    ax.scatter([84],[18],s=10,color=NAVY,zorder=4)
    arrow(ax,(84,18),(59,18),NAVY)
    footer(fig,'Cost-aware scheduling applies to both centroid and PCA certificates. Public routing choices and refinement indices are observable outputs.')
    export(fig,'05_cost_aware_variant','The tested cost-aware engineering variant. Native exact computations only select witness work. A batch activates compact proofs only when predicted savings cover loading/setup and a margin. Every request is retained and publicly verified. This variant is not silently inserted into the existing methodology text.','schematic')


def measured():
    first=read('output/cost_aware_20261029.json')
    final=read('output/cost_aware_final_20261029.json')
    fashion=next(c for c in first['cases'] if c['name']=='fashion_n32')
    text=next(c for c in first['cases'] if c['name']=='text_n32')
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.55))
    fig.subplots_adjust(left=.09,right=.98,bottom=.27,top=.76,wspace=.30)
    fig.suptitle('Measured warm laptop pilots',x=.025,ha='left',y=.98,fontsize=11.4,color=NAVY,weight='bold')
    ax=axes[0]
    vals=[fashion['routed_seconds']/60,fashion['full_seconds']/60]
    bars=ax.bar([0,1],vals,color=[TEAL,GRAY],width=.58)
    ax.set_xticks([0,1],['Certificate','Full']);ax.set_title('Fashion: N=32, Q=8 (paired)',fontsize=8.8)
    ax.set_ylabel('Minutes, fresh warm work');ax.set_ylim(0,11)
    for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.18,f'{v:.2f}',ha='center',fontsize=8)
    ax.text(.5,10.05,f"Full / certificate: {fashion['full_over_routed']:.2f}×",ha='center',color=TEAL,fontsize=8)
    ax=axes[1]
    vals=[text['routed_seconds']/60,text['full_seconds']/60,final['final_text_warm_seconds']/60]
    bars=ax.bar([0,1,2],vals,color=[AMBER,GRAY,BLUE],width=.58)
    bars[2].set_hatch('///')
    ax.set_xticks([0,1,2],['Mixed route\nround 1','Full\nround 1','All-Full\nfinal rerun'])
    ax.set_title('Text: N=32, Q=16',fontsize=8.8);ax.set_ylim(0,11)
    for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.18,f'{v:.2f}',ha='center',fontsize=8)
    ax.text(1,10.05,'Final rerun is not a new timing pair',ha='center',fontsize=6.9,color=GRAY)
    for ax in axes:ax.yaxis.grid(True,color='#DCE3E9',lw=.6);ax.set_axisbelow(True)
    fig.text(.025,.095,'Setup, row authentication and applicable old-cost evidence reused. Planning, fresh query proofs and public replay charged.',fontsize=6.8,color=GRAY)
    fig.text(.025,.038,'One exploratory execution. The final policy preserves the measured image proof plan; text has no demonstrated speed advantage.',fontsize=6.8,color=GRAY)
    export(fig,'06_measured_warm_pilots','Previously measured laptop timings, not new server results. Fashion8 is a paired warm comparison; the final text16 all-Full run is separate from the earlier pair. Setup reuse and measured timing boundaries are explicit. No error bars or broad performance claims are inferred from single executions.','measured exploratory pilot')


def ranks():
    data=read('code/validation/outputs/laptop_four_hour_research_20261028/native_ablations.json')
    assert data['native_only'] and data['new_proofs']==0
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.5))
    fig.subplots_adjust(left=.085,right=.985,bottom=.24,top=.76,wspace=.3)
    fig.suptitle('Certificate rank changes coverage; prediction rank stays fixed',
                 x=.025,ha='left',y=.98,fontsize=10.8,color=NAVY,weight='bold')
    for ax,n in zip(axes,[32,128]):
        for method,name,color,marker in [('ordered_head','Centroid-informed',TEAL,'o'),('pca','PCA',BLUE,'s')]:
            ys=[]
            for rank in [4,39,63,79]:
                records=[r for r in data['records'] if r['dataset']=='fashion' and r['n']==n and
                         r['cap']==16 and r['certificate_rank']==rank and r['method']==method]
                assert len(records)==8
                ys.append(100*sum(r['decision']!='inconclusive' for r in records)/8)
            ax.plot([4,39,63,79],ys,label=name,color=color,marker=marker,
                    ms=5.5 if method=='ordered_head' else 3.5,lw=1.4,
                    ls='-' if method=='ordered_head' else '--',markerfacecolor='white')
        ax.set_title(f'Fashion, N={n}, Q=8',fontsize=8.8);ax.set_xticks([4,39,63,79]);ax.set_ylim(-4,110)
        ax.set_yticks([0,25,50,75,100]);ax.set_xlabel('Certificate rank');ax.grid(axis='y',color='#DCE3E9',lw=.6)
    axes[0].set_ylabel('Conclusive after refinement (%)');axes[1].legend(loc='lower right',fontsize=7,frameon=False)
    fig.text(.025,.092,'Native integer-bound study only: no new proof generation or proof-speed measurement. At most16 corrected rows per request.',fontsize=6.8,color=GRAY)
    fig.text(.025,.035,'Frozen prediction rank63; one previously inspected fit. This is a mechanism diagnostic, not final independent confirmation.',fontsize=6.8,color=GRAY)
    export(fig,'07_rank_coverage_native_only','Existing native integer-bound ablation with fixed prediction rank63 and correction cap16. The plot measures conclusive bound coverage for one exploratory fit, not ZK runtime, formal soundness probability or classification accuracy.','native-only exploratory diagnostic')


def budget():
    warm=read('code/validation/outputs/laptop_four_hour_research_20261028/warm64/checked_comparison.json')
    first=read('output/cost_aware_20261029.json')
    fashion=next(c for c in first['cases'] if c['name']=='fashion_n32')
    records=[dict(method='Centroid warm64',seconds_per_request=warm['measured_warm_certificate_seconds']/64,calibration_q=64),
             dict(method='Full Groth16 warm8',seconds_per_request=fashion['full_seconds']/8,calibration_q=8),
             dict(method='ZKML adapter warm64',seconds_per_request=warm['measured_warm_exact_seconds']/64,calibration_q=64)]
    with (OUT/'request_budget_projection.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['method','calibration_q','seconds_per_request','projected_hours_for_1000','measured_1000','scope'])
        writer.writeheader()
        for r in records:
            writer.writerow(dict(**r,projected_hours_for_1000=r['seconds_per_request']*1000/3600,measured_1000=False,
                scope='Rough laptop-rate planning projection only; not server runtime or measured crossover. Different calibration executions.'))
    return records


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    papers=[REPO/'manuscript.tex',REPO/'references.bib']+sorted((REPO/'paper_results').rglob('*.tex'))
    before={str(p.relative_to(REPO)):sha(p) for p in papers}
    theme();intro();pipeline();refinement();epochs();routing();measured();ranks();rates=budget()
    assert all(sha(REPO/p)==digest for p,digest in before.items())
    (OUT/'captions.md').write_text('# Figure captions and use\n\n'+ '\n\n'.join(
        f"**{r['svg']} — {r['kind']}**\n\n{r['caption']}" for r in FIGURES)+
        '\n\nFigures1–5 are explanatory diagrams. Figures6–7 are existing exploratory evidence, not final server results. '+
        'SVG text remains editable. PNG companions are previews. No manuscript or bibliography was edited.\n',encoding='utf-8')
    (OUT/'manifest.json').write_text(json.dumps(dict(complete=True,figures=FIGURES,sources_sha256=SOURCES,
        generator_sha256=sha(Path(__file__)),paper_sha256=before,paper_unchanged=True,
        svg_text_editable=True,no_new_proofs=True,no_server_jobs=True),indent=2),encoding='utf-8')
    print(json.dumps(dict(complete=True,figures=len(FIGURES),output=str(OUT),budget_rates= rates),indent=2))


if __name__=='__main__':main()
