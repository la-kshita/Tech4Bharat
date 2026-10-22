"""Recount outcomes, verify thresholds and audit the saved split boundaries."""
import argparse
from pathlib import Path

from scipy.stats import binomtest

from component_inspector.data.acquire import sha256
from component_inspector.evaluation.calibration import calibrate
from component_inspector.evaluation.metrics import evaluate
from component_inspector.research.ksdd2_round1 import guard, rows_for, write_json
from component_inspector.research.ksdd2_round2 import context, read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    parent, manifest, _ = context(root)
    guard(manifest['records'])
    role = {r['id']: r['v2_role'] for r in manifest['records']}
    labels = {r['id']: r['label'] for r in rows_for(manifest, 'normal_evaluation')+rows_for(manifest, 'positive_evaluation')}
    checks = {}
    for path in sorted(root.glob('*/result.json')):
        value = read_json(path)
        if 'metrics' not in value:
            for transforms in value.get('models', {}).values():
                for transform in transforms.values():
                    assert evaluate(transform['predictions'], labels) == transform['metrics']
            continue
        predictions, calibration = value['predictions'], value['calibration_scores']
        assert len(predictions) == len(labels) and {r['id'] for r in predictions} == set(labels)
        assert all(r['label'] == labels[r['id']] for r in predictions)
        assert value['manifest_identity'] == manifest['identity']
        assert all(role[r['id']] == 'normal_calibration' for r in calibration)
        assert len({r['id'] for r in calibration}) == len(calibration)
        threshold = calibrate([r['score'] for r in calibration], alpha=.10)
        assert threshold == value['threshold_record']
        assert all(r['flagged'] == (r['score'] > threshold['threshold']) for r in predictions)
        assert evaluate(predictions, labels) == value['metrics']
        details = value.get('details', {})
        if 'model_sha256' in details:
            model_path = path.parent/'head.pt'
            if not model_path.exists():
                name = details['source_model']
                model_path = parent/'supervised/head.pt' if name == 'round1/supervised' else root/name/'head.pt'
            assert sha256(model_path) == details['model_sha256']
        for identity in details.get('fitting_ids', details.get('fit_ids', [])):
            assert role[identity] in {'positive_train', 'normal_fit'}
        if 'normalization_ids' in details:
            assert set(details['normalization_ids']).isdisjoint(details['threshold_ids'])
            assert set(details['threshold_ids']) == {r['id'] for r in calibration}
            assert all(role[i] == 'normal_calibration' for i in details['normalization_ids'])
        if (path.parent/'training.json').exists():
            training = read_json(path.parent/'training.json')
            assert all(role[i] in {'normal_fit', 'positive_train'} for i in training['fitting_ids'])
            assert all(role[i] == 'normal_fit' and i not in manifest['original_fit_ids'] for i in training['hard_normal_ids'])
        checks[path.parent.name] = {'result_sha256': sha256(path), 'recount_and_calibration': 'passed'}
    protocol = read_json(root/'protocol.json')
    for phase in ['aggregation', 'focal', 'lighting_cache', 'lighting', 'fine_cache', 'fine', 'padim50', 'ensemble', 'stress']:
        assert read_json(root/f'{phase}-complete.json')['source_sha256'] == protocol['source_sha256']
    followup = read_json(root/'followup-protocol.json')
    for phase in ['score_heads', 'combined', 'hard', 'fusion_best', 'stability']:
        assert read_json(root/f'{phase}-complete.json')['source_sha256'] == followup['source_sha256']
    for kind in ['fine', 'lighting']:
        index = read_json(root/f'{kind}_features/index.json')['rows']
        assert all(i in role and entry['role'] == role[i] for i, entry in index.items())
        if kind == 'lighting':
            assert all(role[i] in {'normal_fit', 'positive_train'} for i in index)
    selected = read_json(root/'selected-for-stability.json')['selected_result']
    first = read_json(parent/'supervised/result.json')
    second = read_json(root/selected/'result.json')
    a = {r['id']: r for r in first['predictions']}
    b = {r['id']: r for r in second['predictions']}
    discordance = {}
    for label, population in [(1, 'defects'), (0, 'normals')]:
        newly_flagged = [i for i in labels if labels[i] == label and not a[i]['flagged'] and b[i]['flagged']]
        no_longer_flagged = [i for i in labels if labels[i] == label and a[i]['flagged'] and not b[i]['flagged']]
        n = len(newly_flagged)+len(no_longer_flagged)
        discordance[population] = {'newly_flagged_ids': newly_flagged, 'no_longer_flagged_ids': no_longer_flagged,
            'paired_exact_two_sided_p': float(binomtest(len(newly_flagged), n, .5).pvalue) if n else 1.,
            'interpretation': 'descriptive post-selection comparison on observed development; not confirmatory inference'}
    write_json(root/'verification.json', {'manifest_identity': manifest['identity'], 'checked_results': checks,
               'membership': {r: len(rows_for(manifest, r)) for r in sorted(set(role.values()))},
               'selected_result': selected, 'paired_vs_round1_supervised': discordance,
               'historical_test_images_scored_by_these_experiments': False,
               'scope': 'saved record audit plus existing train-only decode guards; no new image decoding in this audit'})
    print('Verified', len(checks), 'results; selected', selected, flush=True)


if __name__ == '__main__':
    main()
