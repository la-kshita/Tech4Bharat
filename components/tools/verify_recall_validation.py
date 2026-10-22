"""Independent saved-row recount and immutable evidence export, without inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import beta, hypergeom


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--export', type=Path, required=True)
    args = parser.parse_args()
    root, target = args.root.resolve(), args.export.resolve()
    p = read(root/'protocol.json')
    summary = read(root/'summary.json')
    predictions = read(root/'predictions.json')
    replay = read(root/'replay.json')
    source = Path(p['parent']).parents[1]/'data/manifests/stage2'/p['historical_manifest_identity']/'manifest.json'
    truth = {r['id']: r['label'] for r in read(source)['splits']['records']
             if r['recipe'] == 'ksdd2_surface' and r['official_split'] == 'test' and not r['quarantine_reason']}
    assert len(truth) == 1004 and sum(truth.values()) == 110
    assert set(p['test_ids']) == set(truth)
    assert p['primary'] == 'warm_aligned' == summary['primary']
    checks = {}
    for name, model in p['models'].items():
        result = read(root/f'{name}-results.json')
        assert len(result['predictions']) == len(predictions['models'][name]) == len(truth)
        saved = {r['id']: r for r in predictions['models'][name]}
        assert set(saved) == set(truth)
        counts = Counter(TP=0, FN=0, FP=0, TN=0, rejected_anomalies=0, rejected_normals=0)
        for row in result['predictions']:
            assert row['label'] == truth[row['id']]
            assert row['status'] == 'scored'
            assert row['score'] == saved[row['id']]['score']
            assert row['flagged'] == saved[row['id']]['flagged'] == (row['score'] > model['threshold'])
            counts[('TP' if row['flagged'] else 'FN') if row['label'] else ('FP' if row['flagged'] else 'TN')] += 1
        assert dict(counts) == result['metrics']['counts'] == summary['outcomes'][name]['metrics']['counts']
        assert result['metrics']['coverage']['rate'] == 1
        tp, fp = counts['TP'], counts['FP']
        low = 0. if not tp else float(beta.ppf(.025, tp, 110-tp+1))
        high = 1. if fp == 894 else float(beta.ppf(.975, fp+1, 894-fp))
        ci = result['confidence']
        assert abs(low-ci['simultaneous_95_recall_lower']) < 1e-10
        assert abs(high-ci['simultaneous_95_fpr_upper']) < 1e-10
        assert ci['point_target_met'] == (tp/110 >= .95 and fp/894 < .1)
        assert ci['confidence_target_met'] == (low >= .95 and high < .1)
        assert sha(Path(model['directory'])/'head.pt') == model['head_sha256']
        assert replay['max_absolute_score_difference'][name] == replay['decision_mismatches'][name] == 0
        checks[name] = {'counts': dict(counts), 'independent_beta_bounds': [low, high],
                        'predictions_and_strict_frozen_threshold_verified': True}
    assert replay['all_scores_bitwise_equal'] and replay['image_count'] == len(truth)
    for file, expected in p['source_sha256'].items():
        assert sha(Path(file)) == expected
    resources = {phase: read(root/f'{phase}-resources.json')
                 for phase in ('integrity', 'infer', 'replay', 'summarize')}
    assert all(r['status'] == 'completed' and r['returncode'] == 0 for r in resources.values())
    primary = read(root/f"{p['primary']}-results.json")
    by = {r['id']: r for r in primary['predictions']}
    counts = primary['metrics']['counts']
    flagged_count = counts['TP']+counts['FP']
    # Conditional random-label control: for the same number of flagged images,
    # how often would a random subset contain at least this many real defects?
    # This is a discrimination diagnostic, not evidence against selection bias.
    chance = float(hypergeom.sf(counts['TP']-1, 1004, 110, flagged_count))
    buckets = {}
    for row in read(source)['splits']['records']:
        if row['id'] not in truth or not truth[row['id']]:
            continue
        image = Path(p['raw_root'])/row['provenance']['source_relative_path']
        mask_path = image.with_name(image.stem+'_GT.png')
        assert sha(mask_path) == row['hashes']['mask_sha256']
        with Image.open(mask_path) as im:
            mask = np.asarray(im.convert('L')) > 0
        fraction = float(mask.mean())
        area = 'area <=0.1%' if fraction <= .001 else 'area (0.1%,0.5%]' if fraction <= .005 else 'area (0.5%,2%]' if fraction <= .02 else 'area >2%'
        edge = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        for key in (area, 'touches source edge' if edge else 'interior defect'):
            bucket = buckets.setdefault(key, {'detected': 0, 'total': 0, 'missed_ids': []})
            bucket['total'] += 1
            bucket['detected'] += int(by[row['id']]['flagged'])
            if not by[row['id']]['flagged']:
                bucket['missed_ids'].append(row['id'])
    for bucket in buckets.values():
        bucket['recall'] = bucket['detected']/bucket['total']
    seeds = ['aligned', 'aligned_seed4218', 'aligned_seed4219']
    flags = {name: {r['id']: r['flagged'] for r in predictions['models'][name]} for name in seeds}
    votes = {identity: sum(flags[name][identity] for name in seeds) for identity in truth}
    seed_consistency = {
        'defects_by_number_of_seeds_detecting': dict(Counter(votes[i] for i in truth if truth[i])),
        'normals_by_number_of_seeds_flagging': dict(Counter(votes[i] for i in truth if not truth[i])),
        'missed_by_all_three_seed_ids': [i for i in truth if truth[i] and votes[i] == 0],
        'scope': 'three cold-start corrected models, all on the same test images; warm continuation is separate',
    }
    write(root/'diagnostics.json', {
        'primary': p['primary'], 'flagged_count': flagged_count,
        'conditional_random_subset_probability_of_at_least_observed_true_positives': chance,
        'random_control_scope': 'rejects random-label discrimination explanation; does not rule out overfitting, benchmark exposure, correlated items, or domain shift',
        'defect_slices': buckets,
        'seed_consistency': seed_consistency,
        'slice_status': 'exploratory after prediction; not preregistered target screens, not used for model selection or threshold changes',
        'source_mask_fraction': 'positive pixels / original full-resolution pixels',
    })
    value = {'protocol_identity': p['identity'], 'independent_recounts': checks,
             'confidence_bounds_independently_checked_with_beta_quantiles': True,
             'source_weights_and_thresholds_unchanged': True,
             'all_resources_completed_within_caps': True,
             'fresh_independent_confirmation_complete': False,
             'verifier_source_sha256': sha(Path(__file__))}
    write(root/'verification.json', value)
    target.mkdir(parents=True, exist_ok=False)
    files = {}
    for path in sorted(root.iterdir()):
        if path.suffix in ('.json', '.log'):
            (target/path.name).write_bytes(path.read_bytes())
            assert sha(target/path.name) == sha(path)
            files[path.name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    write(target/'export-index.json', {'protocol_identity': p['identity'], 'files': files,
                                     'scope': 'exact saved evidence; weights and image arrays remain excluded'})
    print(json.dumps({'verified': checks, 'exported_files': len(files)}, indent=2))


if __name__ == '__main__':
    main()
