"""Read completed server measurements without changing their artifacts."""
from pathlib import Path

from commitjl.calibrated_routing_20261102 import BatchCosts
from experiments.pca_advantage_20261102 import read, sha


def sample_work(sample):
    return (sample.get('native_answer_seconds', 0.) + sample.get('witness_seconds', 0.)
            + sum(sample.get('prove_seconds', [])) + sum(sample.get('verify_seconds', [])))


def calibrate_case(study, name, method):
    study = Path(study)
    paths = {}

    def get(path):
        paths[str(path.relative_to(study))] = sha(path)
        return read(path)

    case = study/'matched'/name
    base = case/(method+'_base')
    ref = case/(method+'_refined')
    baseline = study/'warm'/name/(name+'_ordered_head_warm01')
    full = get(baseline/'baseline_full/private/full_reuse_8.json')
    warm = get(baseline/'checked.json')
    group = get(base/'bundle/group_query.json')
    refinements = get(ref/'bundle/refinement_prove.json')
    bs = get(base/'status.json')['resources']
    rs = get(ref/'status.json')['resources']
    inp = get(base/'input.json')
    full_inp = get(case/method/'full/input.json')
    count = len(inp['reuse_centers'])-1
    assert count == warm['requests'] and count > 0
    assert full['complete'] and group['complete'] and refinements['complete']
    # s000 is the cached old-center cost. It must not be charged as fresh work.
    samples = [s for s in full['samples'] if s.get('centre_set', 0) > 0]
    assert len(samples) == count*len(full_inp['queries'])
    full_body = sum(map(sample_work, samples))
    group_body = sum(map(sample_work, group['samples']))
    ref_body = sum(map(sample_work, refinements['samples']))
    assert refinements['samples'], 'No measured correction profile; do not invent one'
    base_replay = bs['group_verify']['wall_seconds']
    ref_replay = rs['refinement_verify']['wall_seconds']
    full_resources = warm['resources']
    full_wall = full_resources['baseline_full_prove']['wall_seconds']
    full_replay = full_resources['baseline_verify']['wall_seconds']
    costs = BatchCosts(
        full_seconds=full_body/count,
        certificate_seconds=group_body/count,
        correction_seconds=ref_body/len(refinements['samples']),
        full_batch_seconds=max(0., full_wall-full_body)+full_replay,
        certificate_batch_seconds=max(0., bs['group_query']['wall_seconds']-group_body)+base_replay,
        refinement_batch_seconds=max(0., rs['refinement_prove']['wall_seconds']-ref_body)
                                 +max(0., ref_replay-base_replay),
        margin=.10)
    return costs, dict(source=str(study), files=paths, calibration_requests=count,
                       method=method, n=inp['model']['n'], d=inp['model']['d'],
                       rank=len(inp['model']['basis']), input_bits=inp['model']['input_bits'],
                       note='Earlier completed server requests only. Same procedure for both bases. '
                            'A fitted model, not measured latency on new updates.')
