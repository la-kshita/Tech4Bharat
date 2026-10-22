"""Two predeclared outer-refit seeds; neither can replace the primary result."""
import argparse
from pathlib import Path
import subprocess
import sys

import psutil
import torch

from component_inspector.data.acquire import sha256
from component_inspector.evaluation.calibration import calibrate
from component_inspector.research.ksdd2_round1 import digest, read_json, write_json
from component_inspector.research.ksdd2_small_defects import (
    FeatureStore, context, guard, load_selection, partition, result, score, scorer, train_model,
)
from component_inspector.vision.resources import run_bounded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=['register', 'run', 'fit'], required=True)
    parser.add_argument('--fold', type=int, choices=[0, 1, 2])
    parser.add_argument('--offset', type=int, choices=[1000, 2000])
    args = parser.parse_args(); root = args.root.resolve()
    p, _, by = context(root)
    if args.phase == 'register':
        if any(root.glob('outer-*/eval-*.json')):
            raise ValueError('Refit repeats must be registered before any outer result')
        plan = {'protocol_identity': p['identity'], 'source_sha256': sha256(Path(__file__)),
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'seed_offsets': [1000, 2000], 'registered_before_outer_results': True,
                'method_rule': 'same fold-specific method from immutable inner-only selection.json',
                'seed_rule': 'protocol seed +100+outer_fold+offset',
                'partition_rule': 'identical outer fitting/calibration identities and fixed schedule',
                'scope': 'outer-refit initialization/order stability; inner model-selection fits are not repeated',
                'selection_policy': 'report both repeats, never replace primary or final checkpoint using these results',
                'no_official_test_access': True}
        plan['identity'] = digest(plan)
        write_json(root/'refit-repeat-plan.json', plan)
        print('REGISTERED REFIT REPEATS', plan['identity'], flush=True)
        return
    plan = read_json(root/'refit-repeat-plan.json')
    if sha256(Path(__file__)) != plan['source_sha256'] or p['identity'] != plan['protocol_identity']:
        raise ValueError('Refit-repeat source/parent changed')
    if args.phase == 'run':
        if not (root/'nested-results.json').exists(): raise ValueError('Wait for the main study')
        for offset in plan['seed_offsets']:
            for fold in range(3):
                name = f'repeat-{offset}-outer-{fold}'
                resources_file = root/f'{name}-resources.json'
                if resources_file.exists():
                    if read_json(resources_file)['status'] != 'completed':
                        raise RuntimeError('Preserved repeat failure; no blind retry')
                    continue
                cap = min(4*1024**3, int(.60*psutil.virtual_memory().available))
                print('START REFIT REPEAT', offset, fold, 'RSS cap', cap, flush=True)
                resources = run_bounded([sys.executable, str(Path(__file__).resolve()), '--root', str(root),
                    '--phase', 'fit', '--fold', str(fold), '--offset', str(offset)],
                    root/f'{name}-worker.log', max_rss_bytes=cap, max_seconds=1800)
                write_json(resources_file, resources)
                print('END REFIT REPEAT', offset, fold, resources, flush=True)
                if resources['status'] != 'completed': raise RuntimeError('Refit repeat failed')
        return
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    fold, offset = args.fold, args.offset
    selection = load_selection(root, p)
    method = selection['outer_choices'][str(fold)]['method']
    cfg = p['config']['methods'][method]
    fitting, calibration_rows = partition(p, by, [f for f in range(3) if f != fold], salt=f'outer-{fold}')
    guard(fitting+calibration_rows, forbidden_ids={i for i in p['pool_ids'] if p['folds'][i] == fold}
          | set(p['reserved_normal_calibration_ids']))
    directory = root/f'repeat-{offset}-outer-{fold}'; directory.mkdir(exist_ok=False)
    write_json(directory/'plan.json', {'method': method, 'fold': fold, 'offset': offset,
        'selection_identity': selection['identity'], 'repeat_plan_identity': plan['identity'],
        'fitting_ids': [r['id'] for r in fitting], 'calibration_ids': [r['id'] for r in calibration_rows],
        'no_outer_or_test_based_changes': True})
    store = FeatureStore(root, p)
    train_model(directory, store, fitting, cfg, p['config']['seed']+100+fold+offset)
    state = torch.load(directory/'model.pt', weights_only=True); head = scorer(state, cfg)
    calibration = [{'id': r['id'], 'score': score(store, r['id'], cfg, state, head)} for r in calibration_rows]
    threshold = calibrate([r['score'] for r in calibration], alpha=p['config']['calibration_alpha'])
    rows = [by[i] for i in p['pool_ids'] if p['folds'][i] == fold]
    values = {r['id']: score(store, r['id'], cfg, state, head) for r in rows}
    result(directory, f'eval-{fold}.json', rows, values, calibration, threshold, p)
    write_json(directory/'complete.json', {'model_sha256': sha256(directory/'model.pt'),
        'repeat_plan_identity': plan['identity'], 'selection_identity': selection['identity']})


if __name__ == '__main__': main()
