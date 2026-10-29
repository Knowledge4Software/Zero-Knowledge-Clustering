"""Current four-stage methodology: editable SVGs and matching vector PDFs."""
from pathlib import Path
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from figures import server_study_assets_20261029 as a

DEST=Path(__file__).resolve().parents[2]/'paper_results/ccd_pilot_20261029/figures'


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    a.OUT=DEST;a.theme();a.plt.rcParams['pdf.fonttype']=42
    original_export=a.export
    def export(fig,name,caption,kind):
        if name=='03_bounds_refinement':name='methodology_interval_refinement'
        from matplotlib.text import Text
        for label in fig.findobj(match=Text):
            value=label.get_text().replace('most16','most 16').replace('rank63','rank 63')
            label.set_text(value)
        original_export(fig,name,caption,kind)
        fig.savefig(DEST/(name+'.pdf'),metadata={'Title':name,'Subject':caption})
    a.export=export
    fig,ax=a.canvas('Four-stage reusable prototype-update certificate',
         'One deployed reference is authenticated; approved candidate centers can change after preparation.',height=3.6)
    a.label(ax,1,96,'SHARED WITHIN ONE REFERENCE EPISODE',color=a.TEAL,size=7.7,weight='bold')
    a.label(ax,51,96,'REPEATED FOR EACH CANDIDATE',color=a.BLUE,size=7.7,weight='bold')
    a.card(ax,1,49,22,39,'Stage 1: represent',
           'Training-only bases\nPrediction: m = 63\nCertificate: s = 79\nHeads H and energies E',color=a.TEAL,fill=a.GREEN,body_size=6.9)
    a.card(ax,26,49,22,39,'Stage 2: authenticate',
           'Exact old groups gᵢ\nCommit X, H, E, n, S\nPost-commit challenge ξ\nProve every row',color=a.TEAL,fill=a.GREEN,body_size=6.9)
    a.card(ax,51,49,22,39,'Stage 3: bound',
           'Moment link + exact A\nRounded-Gram identity\nSwitching envelope\nL ≤ T⁶Δ ≤ U',body_size=6.9)
    a.card(ax,76,49,23,39,'Stage 4: refine',
           'Select ≤16 exact rows\nProve their corrections\nShrink [L, U]\nStop when conclusive',color=a.AMBER,fill=a.SAND,body_size=6.9)
    for x in [23,48,73]:a.arrow(ax,(x,68),(x+3,68))
    a.card(ax,26,2,47,23,'Verified original-cost decision',
           'Improvement, worsening or equality from verified evidence',color=a.NAVY,fill=a.PALE,body_size=6.7)
    a.card(ax,77,2,22,23,'Exact Full fallback','Generate + verify\nKeep its full cost',color=a.GRAY,fill='#F3F5F7',body_size=6.8)
    a.arrow(ax,(62,49),(62,25),a.BLUE)
    a.label(ax,60,37,'conclusive',ha='right',size=6.7,color=a.BLUE)
    a.arrow(ax,(83,49),(73,25),a.AMBER)
    a.arrow(ax,(91,49),(91,25),a.GRAY)
    a.label(ax,93,36,'unresolved',size=6.3,rotation=90,va='center')
    a.arrow(ax,(77,13),(73,13),a.GRAY)
    a.footer(fig,'Fixed X, C⁰, basis and quantizer within an episode. Original rows remain available for exact corrections and Full.')
    export(fig,'methodology_four_stages',
           'Four stages of the completed methodology. Moment linkage is proved per request with the exact assigned change; all candidate decisions require full authenticated coverage.','methodology schematic')
    a.refinement();a.measured();a.ranks();a.routing()
    data=dict(complete=True,figures=a.FIGURES,sources_sha256=a.SOURCES,
              method_source_sha256=a.sha(DEST.parent/'methodology.tex'),
              generator_sha256=a.sha(Path(__file__)),no_new_experiments=True)
    (DEST/'manifest.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(dict(complete=True,svg_figures=len(a.FIGURES),output=str(DEST))))


if __name__=='__main__':main()
