import json
from experiments.paper_supplement import public_export


def test_public_export_whitelist_and_private_assignment_removal(tmp_path):
    source=tmp_path/'private';source.mkdir()
    for name,value in [('prepare.json',dict(samples=None)),('trusted.json',dict(model={})),
                       ('query_8.json',dict(samples=[dict(assign=[1,2],indices=[0,1],psi='17')]))]:
        (source/name).write_text(json.dumps(value))
    for name in ['input.json','statistics.json','prepared.gob','query_8_progress.json']:
        (source/name).write_text('PRIVATE')
    (source/'keys').mkdir();(source/'keys/query_8.vk').write_bytes(b'PUBLIC')
    (source/'keys/query_8.pk').write_bytes(b'PROVER_KEY')
    target=tmp_path/'public';public_export(source,target)
    assert not any((target/name).exists() for name in ['input.json','statistics.json','prepared.gob','query_8_progress.json','keys/query_8.pk'])
    sample=json.loads((target/'query_8.json').read_text())['samples'][0]
    assert 'assign' not in sample and sample['indices']==[0,1] and sample['psi']=='17'
    assert (target/'keys/query_8.vk').read_bytes()==b'PUBLIC'
