"""Audit every fixed aligned run; never replace canonical seed with a repeat."""
import argparse
from pathlib import Path

import torch

from component_inspector.data.acquire import sha256
from component_inspector.evaluation.calibration import calibrate
from component_inspector.evaluation.metrics import evaluate
from component_inspector.research.ksdd2_round1 import DefectHead, guard, rows_for, write_json
from component_inspector.research.ksdd2_round2 import context, read_json, training_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    _, m, _ = context(root)
    guard(m['records'])
    protocol = read_json(root/'protocol.json')
    by = {r['id']: r for r in m['records']}
    labels = {r['id']: r['label'] for r in rows_for(m, 'normal_evaluation')+rows_for(m, 'positive_evaluation')}
    cal_ids = {r['id'] for r in rows_for(m, 'normal_calibration')}
    base_ids = {r['id'] for r in training_rows(m)}
    outcomes = []
    reference = None
    for seed in [4217, 4218, 4219]:
        method = 'aligned' if seed == 4217 else f'aligned_seed{seed}'
        directory = root/method
        value = read_json(directory/'result.json')
        metadata = read_json(directory/'model.json')
        training = read_json(directory/'training.json')
        assert value['manifest_identity'] == m['identity']
        assert {r['id'] for r in value['calibration_scores']} == cal_ids
        assert len(value['calibration_scores']) == len(cal_ids)
        assert {r['id'] for r in value['predictions']} == set(labels)
        assert len(value['predictions']) == len(labels)
        assert all(p['label'] == labels[p['id']] for p in value['predictions'])
        threshold = calibrate([r['score'] for r in value['calibration_scores']], alpha=.10)
        assert threshold == value['threshold_record']
        assert all(p['flagged'] == (p['score'] > threshold['threshold']) for p in value['predictions'])
        assert evaluate(value['predictions'], labels) == value['metrics']
        assert set(training['base']['fitting_ids']) == base_ids
        hard = set(training['hard_normal_ids'])
        assert len(hard) == len(training['hard_normal_ids']) == 128
        assert all(by[i]['v2_role'] == 'normal_fit' and i not in m['original_fit_ids'] for i in hard)
        assert set(training['refinement']['fitting_ids']) == base_ids|hard
        assert len(training['refinement']['fitting_ids']) == 420
        assert all(by[i]['v2_role'] in {'normal_fit', 'positive_train'} for i in training['fitting_ids'])
        assert training['base']['epochs'] == len(training['base']['history']) == 60
        assert training['refinement']['epochs'] == len(training['refinement']['history']) == 12
        assert not metadata['learned_classifier_used_for_training_or_inference']
        state = torch.load(directory/'head.pt', weights_only=True)
        torch.manual_seed(seed)
        expected = DefectHead(384, 32)
        for key in ['classify.weight', 'classify.bias']:
            assert torch.equal(state['head'][key], expected.state_dict()[key])
        if reference is None:
            reference = state['mean'], state['std']
        assert torch.equal(state['mean'], reference[0]) and torch.equal(state['std'], reference[1])
        assert all(torch.isfinite(v).all() for v in state['head'].values())
        complete = read_json(root/f'{method}-complete.json')
        assert complete['source_sha256'] == protocol['source_sha256']
        assert complete['head_sha256'] == value['details']['model_sha256'] == sha256(directory/'head.pt')
        outcomes.append({'seed': seed, 'method': method, 'metrics': value['metrics'],
                         'result_sha256': sha256(directory/'result.json')})
    passed = all(x['metrics']['recall']['rate'] >= .95 and x['metrics']['accepted_normal_fpr']['rate'] < .10 for x in outcomes)
    warm_audit = None
    if (root/'warm_aligned/result.json').exists():
        plan = read_json(root/'warm-alignment-plan.json')
        directory = root/'warm_aligned'
        value = read_json(directory/'result.json')
        training = read_json(directory/'training.json')
        assert set(training['fitting_ids']) == set(plan['fitting_ids'])
        assert len(training['fitting_ids']) == 420
        assert set(training['hard_normal_ids']) == set(plan['hard_normal_ids'])
        assert all(by[i]['v2_role'] in {'normal_fit', 'positive_train'} for i in training['fitting_ids'])
        assert training['epochs'] == len(training['history']) == plan['epochs'] == 12
        assert {r['id'] for r in value['calibration_scores']} == cal_ids
        assert len(value['calibration_scores']) == len(cal_ids)
        assert {p['id'] for p in value['predictions']} == set(labels)
        assert all(p['label'] == labels[p['id']] for p in value['predictions'])
        threshold = calibrate([r['score'] for r in value['calibration_scores']], alpha=.10)
        assert threshold == value['threshold_record']
        assert all(p['flagged'] == (p['score'] > threshold['threshold']) for p in value['predictions'])
        assert evaluate(value['predictions'], labels) == value['metrics']
        original = torch.load(Path(plan['source_checkpoint'])/'head.pt', weights_only=True)
        state = torch.load(directory/'head.pt', weights_only=True)
        assert torch.equal(state['mean'], original['mean']) and torch.equal(state['std'], original['std'])
        for key in ['classify.weight', 'classify.bias']:
            assert torch.equal(state['head'][key], original['head'][key])
        assert sha256(Path(plan['source_checkpoint'])/'head.pt') == plan['source_sha256']
        assert value['details']['model_sha256'] == sha256(directory/'head.pt')
        assert read_json(root/'warm_aligned-complete.json')['source_sha256'] == plan['worker_source_sha256']
        warm_audit = {'verified': True, 'metrics': value['metrics'], 'result_sha256': sha256(directory/'result.json'),
                      'scope': 'one continuation of a fixed saved checkpoint, not an additional from-scratch repeat'}
    write_json(root/'verification.json', {'all_fixed_seed_records_verified': True,
               'manifest_identity': m['identity'], 'outcomes': outcomes,
               'warm_continuation': warm_audit,
               'all_three_meet_development_point_target': passed,
               'scope': 'same observed split; training-only mining; normal-only calibration; no independent confirmation'})
    write_json(root/'stability-summary.json', {'canonical_seed': 4217, 'canonical_method': 'aligned',
               'outcomes': outcomes, 'all_three_meet_development_point_target': passed,
               'recall_range': [min(x['metrics']['recall']['rate'] for x in outcomes), max(x['metrics']['recall']['rate'] for x in outcomes)],
               'fpr_range': [min(x['metrics']['accepted_normal_fpr']['rate'] for x in outcomes), max(x['metrics']['accepted_normal_fpr']['rate'] for x in outcomes)],
               'claim': 'three fixed initialization/order checks on the same items; not independent data and not three-fold validation'})
    print('Verified all three fixed runs; all meet target:', passed, flush=True)


if __name__ == '__main__':
    main()
