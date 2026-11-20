# Versioned all-source coordinator: compiler resource ceiling only; proof relations unchanged.
import os
from campaigns.server_campaign_20261101 import REPO, GO, RUST, read, save, sha


def warm(campaign, case, repetition):
    from campaigns.server_all_sources_cost_aware_20261101 import calibrate, prove_case
    from commitjl.cost_aware import Planner
    costs, calibration = calibrate(REPO/'server_inputs/calibration')
    out = campaign.root/'warm'/case.name/f'repetition_{repetition:02}'
    marker = out/'checks.json'
    if marker.exists() and read(marker).get('complete'):
        return
    for method in ['ordered_head', 'pca']:
        alias = out/'sources'/method
        alias.mkdir(parents=True, exist_ok=True)
        for role, target in [('ordered_head', case/method), ('ordered_head_base', case/(method+'_base')),
                             ('ordered_head_refined', case/(method+'_refined'))]:
            if not (alias/role).exists():
                (alias/role).symlink_to(target.resolve(), target_is_directory=True)
        prefix = case.name+'_'+method+f'_r{repetition:02}'
        previous = sorted(out.glob(prefix+'_attempt*'))
        if previous and (previous[-1]/'checked.json').exists():
            continue
        # A failed warm trial is retained; a retry gets fresh phase labels so
        # successful pieces of an interrupted trial cannot masquerade as new.
        name = prefix+f'_attempt{len(previous)+1:02}'
        target = out/name
        inp = read(case/(method+'_base/input.json'))
        task = read(case/(method+'_base/approved.json'))
        pins = dict(binary_sha256=sha(GO), certificate=read(case/(method+'_base/owner_pin.json')),
                    full=read(case/method/'full/owner_pin.json'))
        refinement_pin = case/(method+'_refined/owner_refinement_pin.json')
        if refinement_pin.exists():
            pins['refinement'] = read(refinement_pin)
        plan = Planner(inp['model'], inp['x'], costs).batch(task['reuse_centers'])
        save(target/'owner_pin.json', pins)
        save(target/'approved.json', task)
        save(target/'planner.json', plan)
        save(target/'calibration.json', dict(source='Frozen laptop seed521 Q8', sources_sha256=calibration))
        prove_case(campaign, out, dict(name=name, requests=len(task['reuse_centers'])),
                   source=alias, run_baseline=method == 'ordered_head')
    # Fresh Halo2 proofs in a new bundle; immutable setup files may be linked.
    private, public = out/'zkml_private', out/'zkml_public'
    private.mkdir(parents=True, exist_ok=True)
    public.mkdir(exist_ok=True)
    for filename in ['params', 'vk', 'pk', 'owner_pin.json']:
        if not (private/filename).exists():
            os.link(case/'zkml_private'/filename, private/filename)
    for filename in ['params', 'vk']:
        if not (public/filename).exists():
            os.link(private/filename, public/filename)
    source = case/'ordered_head/full/input.json'
    records = []
    for index in range(1, len(read(source)['reuse_centers'])):
        label = case.name+f'_warm_r{repetition:02}_zkml_bank{index}'
        record = campaign.invoke(label, [RUST, 'query', source, private, index])
        destination = public/f'bank_{index:03}'
        destination.mkdir(exist_ok=True)
        for filename in ['proof', 'public.json']:
            shutil.copy2(private/f'bank_{index:03}'/filename, destination/filename)
        replay = campaign.invoke(label+'_verify', [RUST, 'verify', public, index,
                    case/'zkml_owner_pin.json', case/'approved_task.json'])
        assert read(private/f'bank_{index:03}'/'result.json')['cost'] == read(case/'zkml_checked.json')['costs'][index]
        records.append(dict(request=index, generation=record, replay=replay))
    save(marker, dict(complete=True, warm_repetition=repetition, zkml_requests=records,
         timing='Warm process-per-request Halo2 key loading is charged; original reference evidence and immutable setup reused.'))
