"""Frozen-checkpoint evaluation on an already observed official benchmark.

This is a separate access protocol. The older development-only guards remain
unchanged. Test images never enter fitting, calibration, or candidate selection.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
from scipy.stats import binomtest
import torch
from torch.nn import functional as F

from component_inspector.contracts import ImageRecord
from component_inspector.data.acquire import sha256
from component_inspector.data.audit import audit
from component_inspector.evaluation.metrics import evaluate
from component_inspector.research.ksdd2_aligned import fixed_pool
from component_inspector.research.ksdd2_round1 import (
    HISTORICAL, digest, load_manifest, read_json, write_json,
)
from component_inspector.research.ksdd2_round2 import (
    cache_index, cached, model_load, roi_array,
)
from component_inspector.vision.geometry import prepare as prepare_image


METHODS = ('warm_aligned', 'aligned', 'aligned_seed4218', 'aligned_seed4219')
PRIMARY = 'warm_aligned'
EXPECTED_PRIMARY = 'ffca032083bd4c7a003c198b70b7dc5eb10c4d7dee40392fd7a81c9160f560d9'
EXPECTED_THRESHOLD = -1.684261679649353
WEIGHT_HASH = '80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612'
ARCHIVE = HISTORICAL.parents[4]/'historical_benchmark/frozen-20261019T041652Z-9c825801'


def source_files():
    from component_inspector.research import ksdd2_aligned, ksdd2_round1, ksdd2_round2
    from component_inspector.vision import geometry, spike_worker, padim_adapter
    from component_inspector.evaluation import metrics
    from component_inspector.data import audit as audit_module
    modules = [ksdd2_aligned, ksdd2_round1, ksdd2_round2, geometry,
               spike_worker, padim_adapter, metrics, audit_module]
    return [Path(__file__), *[Path(m.__file__) for m in modules]]


def official_rows(raw_root):
    original = read_json(HISTORICAL)
    if digest(original) != HISTORICAL.parent.name:
        raise ValueError('Historical manifest canonical identity changed')
    rows = []
    raw_root = Path(raw_root).resolve()
    for row in original['splits']['records']:
        if row['recipe'] != 'ksdd2_surface' or row.get('quarantine_reason'):
            continue
        relative = Path(row['provenance']['source_relative_path'])
        path = (raw_root/relative).resolve()
        if not path.is_relative_to(raw_root) or relative.parts[0] != row['official_split']:
            raise ValueError('Invalid official source path')
        rows.append({**row, 'path': str(path),
                     'mask_path': str(path.with_name(path.stem+'_GT.png'))})
    return sorted(rows, key=lambda r: r['id'])


def freeze(root, alignment, parent, raw_root, weights):
    root.mkdir(exist_ok=False, parents=True)
    manifest = load_manifest(parent)
    rows = official_rows(raw_root)
    test = [r for r in rows if r['official_split'] == 'test']
    if Counter(r['label'] for r in test) != Counter({0: 894, 1: 110}):
        raise ValueError('Unexpected official test population')
    models = {}
    for name in METHODS:
        directory = alignment/name
        record = read_json(directory/'result.json')
        training = read_json(directory/'training.json')
        metadata = read_json(directory/'model.json')
        head_hash = sha256(directory/'head.pt')
        if (record['manifest_identity'] != manifest['identity']
                or head_hash != record['details']['model_sha256']):
            raise ValueError('Model/manifest mismatch')
        if metadata.get('learned_classifier_used_for_training_or_inference') is not False:
            raise ValueError('Only the corrected fixed-pool candidates are eligible')
        models[name] = {'directory': str(directory), 'head_sha256': head_hash,
                        'threshold': record['threshold_record']['threshold'],
                        'fitting_ids': training['fitting_ids'],
                        'files': {f: sha256(directory/f) for f in
                                  ('head.pt', 'model.json', 'training.json', 'result.json')}}
    if (models[PRIMARY]['head_sha256'] != EXPECTED_PRIMARY
            or models[PRIMARY]['threshold'] != EXPECTED_THRESHOLD):
        raise ValueError('Primary candidate differs from the preselected corrected model')
    if sha256(weights) != WEIGHT_HASH:
        raise ValueError('Backbone weights changed')
    record = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {str(p): sha256(p) for p in source_files()},
        'historical_manifest_identity': HISTORICAL.parent.name,
        'parent': str(parent), 'manifest_identity': manifest['identity'],
        'alignment': str(alignment), 'raw_root': str(raw_root), 'weights': str(weights),
        'weights_sha256': WEIGHT_HASH, 'input': manifest['config']['input'],
        'models': models, 'primary': PRIMARY,
        'test_ids': [r['id'] for r in test],
        'test_population_identity': digest([{k: r[k] for k in ('id', 'hashes', 'group')} for r in test]),
        'archive_sha256': {str(p.relative_to(ARCHIVE)): sha256(p)
                           for p in sorted(ARCHIVE.rglob('*')) if p.is_file()},
        'historical_exposure': 'official test already observed in Stage5; excluded from these heads fitting/calibration but not pristine project confirmation',
        'new_external_data_available': False,
        'rules': {
            'score': 'valid-ROI mean of the top ceil(0.001*N) segmentation logits',
            'features': 'ResNet18 layer2+nearest-layer3; 3x3 mean; float16 roundtrip',
            'comparison': 'strict score > each previously frozen normal-only calibration threshold',
            'point_target': 'recall >= 0.95 AND normal false-positive rate < 0.10 AND complete coverage',
            'uncertainty': 'Clopper-Pearson 95% two-sided intervals; simultaneous one-sided 97.5% bounds for the two primary rates',
            'confidence_screen': 'primary recall lower bound >= .95 AND FPR upper bound < .10; descriptive on this historically observed benchmark',
            'secondary_seeds': 'all three fixed corrected seeds reported; never replace primary using test scores',
            'replay': 'fresh process, reverse image order, all 1004 images and all four checkpoints; exact score/decision comparison',
            'fit_diagnostic': 'score each models existing 420 fitting images using existing cache; descriptive only',
            'no_training_or_recalibration': True,
            'failure_policy': 'abort on integrity error, incomplete scores, or model/source mutation; preserve failures; no tuning until pass',
            'independence': 'physical product/production lot identifiers unavailable; appearance auditing cannot prove independence',
        },
    }
    record['identity'] = digest(record)
    write_json(root/'protocol.json', record)
    print('FROZEN', record['identity'], PRIMARY, EXPECTED_THRESHOLD, flush=True)


def context(root):
    p = read_json(root/'protocol.json')
    identity = p.pop('identity')
    if digest(p) != identity:
        raise ValueError('Protocol identity changed')
    p['identity'] = identity
    for file, expected in p['source_sha256'].items():
        if sha256(file) != expected:
            raise ValueError(f'Frozen inference/audit source changed: {file}')
    if sha256(p['weights']) != p['weights_sha256']:
        raise ValueError('Frozen backbone changed')
    for model in p['models'].values():
        for file, expected in model['files'].items():
            if sha256(Path(model['directory'])/file) != expected:
                raise ValueError('Frozen model artifact changed')
    m = load_manifest(Path(p['parent']))
    if m['identity'] != p['manifest_identity']:
        raise ValueError('Development membership changed')
    rows = official_rows(p['raw_root'])
    test = [r for r in rows if r['official_split'] == 'test']
    if ([r['id'] for r in test] != p['test_ids'] or
            digest([{k: r[k] for k in ('id', 'hashes', 'group')} for r in test]) != p['test_population_identity']):
        raise ValueError('Test membership changed')
    return p, m, rows, test


def overlap(a, b, key):
    return sorted({r[key] if key in r else r['hashes'][key] for r in a}
                  & {r[key] if key in r else r['hashes'][key] for r in b})


def integrity(root):
    p, m, rows, test = context(root)
    fresh = audit([ImageRecord(**r) for r in rows])
    by = {r['id']: r for r in rows}
    for r in fresh.records:
        if r.quarantine_reason or r.hashes != by[r.id]['hashes'] or r.group != by[r.id]['group']:
            raise ValueError(f'Raw release/hash/mask/duplicate audit mismatch: {r.id}')
    train = [r for r in rows if r['official_split'] == 'train']
    overlaps = {key: overlap(train, test, key) for key in
                ('id', 'group', 'sha256', 'decoded_sha256')}
    if any(overlaps.values()):
        raise ValueError('Official test overlaps training originals')
    cal_ids = {r['id'] for r in m['records'] if r['v2_role'] == 'normal_calibration'}
    eval_ids = {r['id'] for r in m['records'] if r['v2_role'].endswith('evaluation')}
    by_train = {r['id']: r for r in m['records']}
    usage = {}
    for name, model in p['models'].items():
        fit = model['fitting_ids']
        if (len(fit) != 420 or len(set(fit)) != 420 or set(fit) & (cal_ids | eval_ids)
                or any(i not in by_train or by_train[i]['v2_role'] not in
                       ('positive_train', 'normal_fit') for i in fit)):
            raise ValueError('Invalid fitting membership')
        result = read_json(Path(model['directory'])/'result.json')
        if {r['id'] for r in result['calibration_scores']} != cal_ids:
            raise ValueError('Calibration membership differs')
        # Verify the saved threshold was fixed using the original normal-only
        # calibration scores, never this official test set.
        from component_inspector.evaluation.calibration import calibrate
        threshold = calibrate([r['score'] for r in result['calibration_scores']], alpha=.1)
        if threshold != result['threshold_record'] or threshold['threshold'] != model['threshold']:
            raise ValueError('Frozen threshold does not reproduce')
        usage[name] = {'fitting_counts': dict(Counter(by_train[i]['v2_role'] for i in fit)),
                       'normal_calibration_count': len(cal_ids),
                       'fitting_calibration_or_development_overlap': 0,
                       'official_test_in_training_or_calibration': 0}
    write_json(root/'integrity.json', {
        'protocol_identity': p['identity'], 'raw_image_and_mask_hashes_verified': len(rows),
        'mask_binary_shape_and_label_checks': 'all passed',
        'all_original_labels': dict(Counter(f"{r['official_split']}/{r['label']}" for r in rows)),
        'train_test_overlaps': overlaps, 'fresh_appearance_audit': fresh.summary,
        'duplicate_pairs': fresh.duplicate_pairs, 'usage': usage,
        'near_duplicate_rule': 'pHash Hamming <=2 AND 64x64 RGB thumbnail MAE <=2/255; conservative screen, not identity proof',
        'physical_product_or_lot_independence': 'unknown',
        'historical_exposure': p['historical_exposure'],
    })
    print('INTEGRITY PASS', len(rows), 'originals;', len(test), 'test images', flush=True)


def infer_one(adapter, path, cfg, models):
    # No mask, label, ID, filename features, or historical score reaches the net.
    with Image.open(path) as im:
        prepared = prepare_image(im, cfg)
    f = adapter.model.feature_extractor(adapter._tensor(prepared.rgb_float32))
    f2, f3 = f['layer2'], f['layer3']
    embedding = F.avg_pool2d(torch.cat([
        f2, F.interpolate(f3, size=f2.shape[-2:], mode='nearest')], 1),
        3, stride=1, padding=1).half().float()
    roi = F.interpolate(torch.from_numpy(prepared.valid_roi.astype(np.float32))[None, None],
                        size=embedding.shape[-2:], mode='nearest')[:, 0].bool()
    return head_scores(embedding, roi, models)


def head_scores(embedding, roi, models):
    scores = {}
    for name, (head, state, _) in models.items():
        raw = head.segment((embedding-state['mean'])/state['std'])[:, 0]
        value = float(fixed_pool(raw, roi)[0])
        if not np.isfinite(value):
            raise ValueError('Nonfinite score')
        scores[name] = value
    return scores


@torch.inference_mode()
def inference(root, replay=False):
    from component_inspector.vision.spike_worker import pretrained_adapter
    p, m, _, test = context(root)
    check = read_json(root/'integrity.json')
    if check['protocol_identity'] != p['identity']:
        raise ValueError('Missing matching integrity audit')
    models = {name: model_load(Path(info['directory'])) for name, info in p['models'].items()}
    adapter = pretrained_adapter('ksdd2_surface', m['config'], Path(p['weights']))
    before = {name: {k: v.clone() for k, v in head.state_dict().items()}
              for name, (head, _, _) in models.items()}
    expected = read_json(root/'predictions.json')['models'] if replay else None
    expected = {name: {r['id']: r for r in rows} for name, rows in expected.items()} if replay else None
    predictions, times = {name: [] for name in models}, []
    differences, mismatches = {name: 0. for name in models}, {name: 0 for name in models}
    for i, row in enumerate(reversed(test) if replay else test):
        tick = time.perf_counter()
        if sha256(row['path']) != row['hashes']['sha256']:
            raise ValueError('Input bytes changed')
        values = infer_one(adapter, row['path'], m['config'], models)
        times.append(time.perf_counter()-tick)
        for name, score in values.items():
            flagged = score > p['models'][name]['threshold']
            predictions[name].append({'id': row['id'], 'score': score, 'flagged': flagged, 'status': 'scored'})
            if replay:
                prior = expected[name][row['id']]
                differences[name] = max(differences[name], abs(score-prior['score']))
                mismatches[name] += int(flagged != prior['flagged'])
        if i % 100 == 0 or i+1 == len(test):
            print('REPLAY' if replay else 'INFERENCE', i+1, '/', len(test), flush=True)
    for name, (head, _, _) in models.items():
        if any(not torch.equal(v, before[name][k]) for k, v in head.state_dict().items()):
            raise ValueError('Evaluation changed head parameters or buffers')
    context(root)  # Verify frozen files again after the full image pass.
    record = {'protocol_identity': p['identity'], 'image_count': len(test),
              'pipeline_seconds': times, 'p95_seconds': float(np.percentile(times, 95)),
              'no_parameter_or_buffer_mutation': True}
    if replay:
        record.update(max_absolute_score_difference=differences, decision_mismatches=mismatches,
                      all_scores_bitwise_equal=all(v == 0 for v in differences.values()),
                      scope='fresh process and fresh image features, reverse order, all models; repeated observations do not increase sample size')
        write_json(root/'replay.json', record)
        if any(differences.values()) or any(mismatches.values()):
            raise ValueError('Full replay was not exactly reproducible')
    else:
        write_json(root/'predictions.json', {**record, 'models': predictions,
                   'no_ground_truth_in_prediction_records': True})


def confidence(metrics):
    tp, n = metrics['recall']['numerator'], metrics['recall']['denominator']
    fp, normals = metrics['false_flags_all_normals']['numerator'], metrics['false_flags_all_normals']['denominator']
    if not n or not normals or metrics['coverage']['rate'] != 1:
        raise ValueError('Confidence screen requires both classes and complete coverage')
    recall_ci = binomtest(tp, n).proportion_ci(method='exact')
    fpr_ci = binomtest(fp, normals).proportion_ci(method='exact')
    # Bonferroni: two one-sided bounds with alpha .025 each, at least 95%
    # simultaneous coverage under independent binomial sampling assumptions.
    lower = float(binomtest(tp, n, alternative='greater').proportion_ci(.975, method='exact').low)
    upper = float(binomtest(fp, normals, alternative='less').proportion_ci(.975, method='exact').high)
    return {'recall_exact_95': [float(recall_ci.low), float(recall_ci.high)],
            'fpr_exact_95': [float(fpr_ci.low), float(fpr_ci.high)],
            'simultaneous_95_recall_lower': lower, 'simultaneous_95_fpr_upper': upper,
            'point_target_met': tp/n >= .95 and fp/normals < .1,
            'confidence_target_met': lower >= .95 and upper < .1,
            'assumptions': 'iid representative items, fixed model and threshold; observed benchmark and unknown physical groups limit interpretation'}


@torch.inference_mode()
def summarize(root):
    p, m, _, test = context(root)
    record = read_json(root/'predictions.json')
    replay = read_json(root/'replay.json')
    if record['protocol_identity'] != p['identity'] or not replay['all_scores_bitwise_equal']:
        raise ValueError('Prediction protocol or replay mismatch')
    truth = {r['id']: r['label'] for r in test}
    base = Path(p['parent'])/'features'
    index = cache_index(base)['rows']
    by_train = {r['id']: r for r in m['records']}
    outcomes = {}
    for name, model in p['models'].items():
        predictions = record['models'][name]
        if len(predictions) != len(test) or {r['id'] for r in predictions} != set(truth):
            raise ValueError('Incomplete or duplicate test membership')
        labeled = [{**r, 'label': truth[r['id']]} for r in predictions]
        metrics = evaluate(predictions, truth)
        head = {name: model_load(Path(model['directory']))}
        fitted_predictions = []
        for identity in model['fitting_ids']:
            info = index[identity]
            feature = torch.from_numpy(cached(base, info))[None]
            roi = torch.from_numpy(roi_array(base, info))[None]
            score = head_scores(feature, roi, head)[name]
            fitted_predictions.append({'id': identity, 'score': score,
                                       'flagged': score > model['threshold'], 'status': 'scored'})
        fitting_truth = {i: by_train[i]['label'] for i in model['fitting_ids']}
        fitting = evaluate(fitted_predictions, fitting_truth)
        development = read_json(Path(model['directory'])/'result.json')['metrics']
        outcomes[name] = {'metrics': metrics, 'confidence': confidence(metrics),
                          'threshold': model['threshold'], 'head_sha256': model['head_sha256'],
                          'training_diagnostic': fitting, 'development_metrics': development,
                          'test_minus_development_recall': metrics['recall']['rate']-development['recall']['rate'],
                          'test_minus_development_fpr': metrics['accepted_normal_fpr']['rate']-development['accepted_normal_fpr']['rate']}
        write_json(root/f'{name}-results.json', {
            'protocol_identity': p['identity'], **outcomes[name], 'predictions': labeled,
            'training_predictions_diagnostic_only': fitted_predictions,
            'false_negative_ids': [r['id'] for r in labeled if r['label'] and not r['flagged']],
            'false_positive_ids': [r['id'] for r in labeled if not r['label'] and r['flagged']]})
        print(name, metrics['counts'], outcomes[name]['confidence'], flush=True)
    archive_changed = [f for f, expected in p['archive_sha256'].items() if sha256(ARCHIVE/f) != expected]
    if archive_changed:
        raise ValueError('Historical archive changed')
    write_json(root/'summary.json', {
        'protocol_identity': p['identity'], 'primary': PRIMARY, 'outcomes': outcomes,
        'fresh_independent_validation_complete': False,
        'reason': p['historical_exposure']+'; user confirms no separate new labeled batch',
        'test_images': len(test), 'test_defects': sum(truth.values()),
        'test_normals': len(test)-sum(truth.values()),
        'all_four_models_full_replay_exact': replay['all_scores_bitwise_equal'],
        'historical_archive_files_unchanged': len(p['archive_sha256']),
        'no_training_threshold_or_candidate_changes': True,
        'secondary_results_are_not_new_candidate_selection': True,
        'no_absolute_certainty_or_factory_generalization_claim': True,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['freeze', 'integrity', 'infer', 'replay', 'summarize'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--alignment', type=Path, default=Path('components/outputs/recall-alignment'))
    parser.add_argument('--parent', type=Path, default=Path('components/outputs/recall-round1'))
    parser.add_argument('--raw-root', type=Path, default=Path('components/outputs/recall-data/data/raw/ksdd2'))
    parser.add_argument('--weights', type=Path, default=Path('.cache/recall-downloads/model.safetensors'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.phase == 'freeze':
        freeze(args.root.resolve(), args.alignment.resolve(), args.parent.resolve(),
               args.raw_root.resolve(), args.weights.resolve())
    elif args.phase == 'integrity':
        integrity(args.root.resolve())
    elif args.phase in ('infer', 'replay'):
        inference(args.root.resolve(), replay=args.phase == 'replay')
    else:
        summarize(args.root.resolve())


if __name__ == '__main__':
    main()
