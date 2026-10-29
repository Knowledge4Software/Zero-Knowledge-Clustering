"""Check the private server packet before compiling or running anything."""
from pathlib import Path
import hashlib,json


def validate(repo):
    manifest=json.loads((repo/'round3_packet_manifest.json').read_text())
    for name,expected in manifest.items():
        path=(repo/name).resolve()
        if not path.is_relative_to(repo.resolve()):raise ValueError('Path outside packet: '+name)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Changed or missing packet file: '+name)
    return dict(complete=True,files_checked=len(manifest),proofs_generated=False)


if __name__=='__main__':print(json.dumps(validate(Path(__file__).resolve().parents[2])))
