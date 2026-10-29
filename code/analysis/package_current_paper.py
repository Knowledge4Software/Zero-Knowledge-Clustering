"""Package current paper dependencies and numerical provenance for private review.

Compile manuscript.tex first. This does not certify uncaptured server results.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import zipfile

from build_three_papers import ROOT, expand


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='neurocomputing_referee_20261023')
    tag = parser.parse_args().tag
    if not tag or Path(tag).name != tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in tag):
        raise ValueError('tag must be a simple filename stem')
    files = set()
    expand(ROOT / 'manuscript.tex', files)
    files.add(ROOT / 'references.bib')
    for folder in ['paired_20261022', 'scale_quality_native_20261023', 'referee_20261023']:
        files.update(p for p in (ROOT / 'paper_results/review_controls' / folder).rglob('*')
                     if p.is_file() and p.suffix in {'.json', '.csv'})
    files.add(ROOT / 'code/docs/review_handoff_20261023.md')
    for name in ['code/analysis/analyze_natural_regressions.py', 'code/analysis/check_family_affine.py',
                 'code/commitjl/family_affine.py', 'code/docs/referee_response_20261023.md']:
        files.add(ROOT/name)
    data = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted(files)}
    for name in ['elsarticle.cls', 'elsarticle-num.bst']:
        path = Path(subprocess.check_output(['kpsewhich', name], text=True).strip())
        data[name] = path.read_bytes()
    data['README_RELEASE.md'] = (
        '# Private supervisor-review paper, 23 October 2026\n\n'
        'Compile: latexmk -pdf manuscript.tex\n\n'
        'This package contains the actual current TeX dependencies and selected\n'
        'numerical provenance. It is not a complete proof-reproduction artifact.\n'
        'Captured tables retain the 22 October cutoff. New scale timings are\n'
        'console transcriptions; quality comes from local SHA-pinned inputs.\n'
        'Original completed server outputs and independent replay remain pending.\n'
        'No public release is authorized. Supervisor finalizes author fields.\n'
    ).encode()
    manifest = [{'path': k, 'sha256': hashlib.sha256(v).hexdigest()}
                for k, v in sorted(data.items())]
    data['source_manifest.json'] = json.dumps(manifest, indent=2).encode()
    target = ROOT / 'output' / (tag + '_sources.zip')
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, value in sorted(data.items()):
            z.writestr(name, value)
    with zipfile.ZipFile(target) as z:
        for item in manifest:
            assert hashlib.sha256(z.read(item['path'])).hexdigest() == item['sha256']
    print(json.dumps({'archive': str(target), 'files': len(data),
                      'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                      'archive_hash_checks': 'PASS'}, indent=2))


if __name__ == '__main__':
    main()
