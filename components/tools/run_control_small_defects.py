"""Predeclared paired outer comparator; never participates in model selection."""
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
    FeatureStore, context, guard, partition, result, score, scorer, train_model,
)
from component_inspector.vision.resources import run_bounded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=['register', 'run', 'fit'], required=True)
    parser.add_argument('--fold', type=int, choices=[0, 1, 2])
    args = parser.parse_args(); root = args.root.resolve()
    p, _, by = context(root)
    if args.phase == 'register':
        if any(root.glob('outer-*/eval-*.json')):
            raise ValueError('Paired control must be registered before any outer result')
        plan = {'protocol_identity': p['identity'], 'method': 'coarse_control',
                'source_sha256': sha256(Path(__file__)),
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'configuration': p['config']['methods']['coarse_control'],
                'folds': [0, 1, 2], 'registered_before_outer_results': True,
                'fit_seed_rule': 'same as selected outer model: protocol seed +100+fold',
                'partition_rule': 'same outer-{fold} salt and identical fitting/calibration identities',
                'role': 'fixed paired comparator only; never change selection or final candidate',
                'no_official_test_access': True}
        plan['identity'] = digest(plan)
        write_json(root/'paired-control-plan.json', plan)
        print('REGISTERED FIXED PAIRED CONTROL', plan['identity'], flush=True)
        return
    plan = read_json(root/'paired-control-plan.json')
    if sha256(Path(__file__)) != plan['source_sha256'] or plan['protocol_identity'] != p['identity']:
        raise ValueError('Paired-control source or parent changed')
    if args.phase == 'run':
        if not (root/'nested-results.json').exists():
            raise ValueError('Run paired control after the unchanged main study completes')
        for fold in range(3):
            resources_file = root/f'control-outer-{fold}-resources.json'
            if resources_file.exists():
                if read_json(resources_file)['status'] != 'completed':
                    raise RuntimeError('Preserved control failure; do not blindly repeat')
                continue
            cap = min(4*1024**3, int(.60*psutil.virtual_memory().available))
            print('START FIXED CONTROL', fold, 'RSS cap', cap, flush=True)
            resources = run_bounded([sys.executable, str(Path(__file__).resolve()), '--root', str(root),
                                     '--phase', 'fit', '--fold', str(fold)],
                                     root/f'control-outer-{fold}-worker.log', max_rss_bytes=cap, max_seconds=1800)
            write_json(resources_file, resources)
            print('END FIXED CONTROL', fold, resources, flush=True)
            if resources['status'] != 'completed': raise RuntimeError('Fixed control failed')
        return
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    fold = args.fold
    train_folds = [f for f in range(3) if f != fold]
    fitting, calibration_rows = partition(p, by, train_folds, salt=f'outer-{fold}')
    forbidden = {i for i in p['pool_ids'] if p['folds'][i] == fold}
    guard(fitting+calibration_rows, forbidden_ids=forbidden | set(p['reserved_normal_calibration_ids']))
    directory = root/f'control-outer-{fold}'; directory.mkdir(exist_ok=False)
    write_json(directory/'plan.json', {'stage': 'fixed_outer_control', 'fold': fold, 'method': 'coarse_control',
               'fitting_ids': [r['id'] for r in fitting], 'calibration_ids': [r['id'] for r in calibration_rows],
               'excluded_fold_ids': sorted(forbidden), 'paired_control_identity': plan['identity'],
               'official_test_access': False, 'not_used_in_selection': True})
    cfg = plan['configuration']; store = FeatureStore(root, p)
    train_model(directory, store, fitting, cfg, p['config']['seed']+100+fold)
    state = torch.load(directory/'model.pt', weights_only=True); head = scorer(state, cfg)
    calibration = [{'id': r['id'], 'score': score(store, r['id'], cfg, state, head)} for r in calibration_rows]
    threshold = calibrate([r['score'] for r in calibration], alpha=p['config']['calibration_alpha'])
    rows = [by[i] for i in p['pool_ids'] if p['folds'][i] == fold]
    values = {r['id']: score(store, r['id'], cfg, state, head) for r in rows}
    result(directory, f'eval-{fold}.json', rows, values, calibration, threshold, p)
    write_json(directory/'complete.json', {'model_sha256': sha256(directory/'model.pt'),
               'paired_control_identity': plan['identity'], 'protocol_identity': p['identity']})


if __name__ == '__main__': main()
