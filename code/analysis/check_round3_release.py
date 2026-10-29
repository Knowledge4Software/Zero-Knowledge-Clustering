"""Lightweight integrity/layout comparison after the presentation-only revision."""
from pathlib import Path
import argparse,hashlib,json,re,subprocess
from campaigns.local_fast_tradeoff import save,sha

REPO=Path(__file__).resolve().parents[2]
TAG='prototype_audit_referee_revision_20261026'
BASE_COMMIT='297328a'
REVIEWED_RENDER=REPO/'tmp/pdfs'/TAG/'renders/5839044190c1'


def numeric_cells(text):
    return [cell.strip().removesuffix('\\\\').strip() for line in text.splitlines() if '&' in line
        for cell in line.split('&') if re.fullmatch(r'[\d.,/%+\-]+',cell.strip().removesuffix('\\\\').strip())]


def main(mark=False):
    path=REPO/'output'/(TAG+'_validation.json');v=json.loads(path.read_text());assert sha(REPO/v['pdf'])==v['pdf_sha256']
    tables=REPO/'paper_results/referee_final_20261025/generated';checked=[]
    for table in tables.glob('*.tex'):
        relative=table.relative_to(REPO).as_posix()
        before=subprocess.check_output(['git','show',BASE_COMMIT+':'+relative],cwd=REPO,text=True)
        assert numeric_cells(before)==numeric_cells(table.read_text()),'Changed numerical cells: '+relative
        checked.append(relative)
    current=REPO/v['render_directory'];same=[];changed=[]
    for image in sorted(current.glob('page-*.png')):
        (same if sha(image)==sha(REVIEWED_RENDER/image.name) else changed).append(image.name)
    check=dict(complete=True,numeric_tables_checked=len(checked),numeric_cells_unchanged=True,baseline_commit=BASE_COMMIT,
        pdf_sha256=v['pdf_sha256'],unchanged_previously_inspected_pages=same,changed_pages=changed,
        prior_all_pages_visually_reviewed=True,changed_pages_visual_review='PASS' if mark else 'PENDING')
    save(REPO/'output/referee_round3_release_checks_20261026.json',check)
    if mark:
        assert changed==['page-08.png'],'Inspect any additional changed pages before approving'
        v.update(status='ready_for_private_supervisor_review',visual_review='PASS',visual_review_pages=list(range(1,v['pages']+1)),
            visual_review_method='All 14 prior rendered pages inspected. Final render has 13 byte-identical pages and one changed table page, inspected again.',
            unresolved_scientific_issues=True,submission_ready=False)
        save(path,v)
    print(json.dumps(check,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mark-visual-reviewed',action='store_true');a=p.parse_args();main(a.mark_visual_reviewed)
