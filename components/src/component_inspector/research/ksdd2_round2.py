"""Bounded continuation of the observed KSDD2 development experiments.

R01 artifacts and membership stay immutable. No official test images enter any
phase. Normal-only calibration remains separate from fitting and refinement.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch import nn
from torch.nn import functional as F

from component_inspector.data.acquire import sha256
from component_inspector.evaluation.calibration import calibrate
from component_inspector.evaluation.metrics import evaluate
from component_inspector.research.ksdd2_round1 import (
    PROJECT, DefectHead, digest, gaussian_by_position, guard, image_array,
    load_feature, load_manifest, read_json, rows_for, write_json,
)


def context(root):
    protocol = read_json(root/'protocol.json')
    parent = Path(protocol['parent']).resolve()
    m = load_manifest(parent)
    if m['identity'] != protocol['manifest_identity']:
        raise ValueError('Parent membership changed')
    return parent, m, protocol['config']


def cache_index(path):
    return read_json(path/'index.json')


def cached(path, info):
    return np.load(path/f"{info['key']}.npy", allow_pickle=False).astype(np.float32)


def roi_array(path, info):
    return np.load(path/f"{info['key']}-roi.npy", allow_pickle=False)


def model_load(path):
    metadata = read_json(path/'model.json') if (path/'model.json').exists() else {'channels': 384, 'hidden': 32}
    state = torch.load(path/'head.pt', weights_only=True, map_location='cpu')
    head = DefectHead(metadata['channels'], metadata['hidden'])
    head.load_state_dict(state['head'])
    head.eval()
    return head, state, metadata


def result(root, name, m, scores, calibration, *, details=None):
    evaluation = rows_for(m, 'normal_evaluation')+rows_for(m, 'positive_evaluation')
    if set(scores) != {r['id'] for r in evaluation}:
        raise ValueError('Wrong or incomplete evaluation membership')
    expected_cal = {r['id'] for r in rows_for(m, 'normal_calibration')}
    if not set(calibration).issubset(expected_cal) or len(calibration) < 9:
        raise ValueError('Invalid normal calibration membership')
    threshold = calibrate(list(calibration.values()), alpha=.10)
    predictions = [{'id': r['id'], 'group': r['group'], 'label': r['label'],
                    'status': 'scored', 'score': float(scores[r['id']]),
                    'flagged': bool(scores[r['id']] > threshold['threshold'])} for r in evaluation]
    value = {'method': name, 'claim': 'observed single-split development comparison; no independent validation',
             'manifest_identity': m['identity'], 'threshold_record': threshold,
             'metrics': evaluate(predictions, {r['id']: r['label'] for r in evaluation}),
             'predictions': predictions,
             'calibration_scores': [{'id': i, 'score': float(s)} for i, s in sorted(calibration.items())],
             'details': details or {}}
    write_json(root/name/'result.json', value)
    print(json.dumps({'method': name, 'recall': value['metrics']['recall']['rate'],
                      'fpr': value['metrics']['accepted_normal_fpr']['rate'],
                      'counts': value['metrics']['counts']}), flush=True)
    return value


def score_rules(raw, roi):
    valid = raw[roi]
    if not len(valid) or not torch.isfinite(valid).all():
        raise ValueError('Invalid scoring map')
    k = max(1, math.ceil(.001*len(valid)))
    smooth = F.avg_pool2d((raw*roi)[None, None], 3, stride=1, padding=1)[0, 0]
    count = F.avg_pool2d(roi.float()[None, None], 3, stride=1, padding=1)[0, 0]
    local = smooth/count.clamp_min(1e-8)
    return {'maximum': float(valid.max()), 'top_0.1_percent': float(valid.topk(k).values.mean()),
            'maximum_local_3x3_average': float(local[roi].max())}


@torch.inference_mode()
def aggregation(root):
    parent, m, cfg = context(root)
    index = cache_index(parent/'features')['rows']
    head, state, _ = model_load(parent/'supervised')
    score = {rule: {} for rule in cfg['aggregation']}
    cal = {rule: {} for rule in cfg['aggregation']}
    peaks = []
    for role in ['normal_calibration', 'normal_evaluation', 'positive_evaluation']:
        for row in rows_for(m, role):
            entry = index[row['id']]
            f = torch.from_numpy(cached(parent/'features', entry))[None]
            roi = torch.from_numpy(roi_array(parent/'features', entry))
            raw, _ = head((f-state['mean'])/state['std'], roi[None])
            values = score_rules(raw[0], roi)
            for rule in values:
                (cal if role == 'normal_calibration' else score)[rule][row['id']] = values[rule]
            if role == 'normal_evaluation':
                masked = raw[0].masked_fill(~roi, -torch.inf)
                y, x = np.unravel_index(int(masked.argmax()), tuple(masked.shape))
                ys, xs = torch.where(roi)
                border = bool(y < int(ys.min())+2 or y > int(ys.max())-2
                              or x < int(xs.min())+2 or x > int(xs.max())-2)
                peaks.append({'id': row['id'], 'peak_y': int(y), 'peak_x': int(x), 'outer_two_cells': border})
    for rule in cfg['aggregation']:
        result(root, 'aggregation_'+rule, m, score[rule], cal[rule],
               details={'strategy': 'R06', 'parent': 'round1/supervised', 'rule': rule,
                        'no_refitting': True, 'input': 'same parent segmentation map; no defect masks used in inference'})
    write_json(root/'roi-diagnostic.json', {'strategy': 'R07', 'status': 'deferred_with_reason',
               'reason': 'KSDD2 is a surface texture; no established removable object background or pose ambiguity. Padding is already excluded. Foreground masking/rigid alignment could remove edge defects; no such transform is justified by this task.',
               'normal_peak_outer_two_cell_fraction': sum(r['outer_two_cells'] for r in peaks)/len(peaks),
               'normal_peak_records': peaks, 'official_test_access': False})


def training_rows(m):
    by = {r['id']: r for r in m['records']}
    ids = sorted(m['original_fit_ids'])+[r['id'] for r in rows_for(m, 'positive_train')]
    selected = [by[i] for i in ids]
    guard(selected)
    assert all(r['v2_role'] in ['normal_fit', 'positive_train'] for r in selected)
    return selected


@torch.inference_mode()
def make_cache(root, kind, weights):
    from component_inspector.vision.spike_worker import pretrained_adapter
    parent, m, cfg = context(root)
    adapter = pretrained_adapter('ksdd2_surface', m['config'], weights)
    directory = root/(kind+'_features')
    directory.mkdir(exist_ok=False)
    if kind == 'lighting':
        selected = training_rows(m)
    else:
        wanted = set(m['original_fit_ids'])
        selected = sorted((r for r in m['records'] if r['id'] in wanted or r['v2_role'] != 'normal_fit'), key=lambda r: r['id'])
    generator = torch.Generator().manual_seed(cfg['fine_features']['projection_seed'])
    projection = torch.randn(cfg['fine_features']['projection_channels'], 448, 1, 1, generator=generator)/math.sqrt(448)
    rng = np.random.default_rng(cfg['seed'])
    index = {}
    for i, row in enumerate(selected):
        prepared = image_array(row, m['config'])
        transforms = None
        if kind == 'lighting':
            brightness = float(rng.uniform(*cfg['lighting']['brightness']))
            contrast = float(rng.uniform(*cfg['lighting']['contrast']))
            with Image.open(row['path']) as im:
                im = ImageEnhance.Brightness(im.convert('RGB')).enhance(brightness)
                im = ImageEnhance.Contrast(im).enhance(contrast)
                from component_inspector.vision.geometry import prepare
                prepared = prepare(im, m['config'])
            transforms = {'brightness': brightness, 'contrast': contrast}
        tensors = adapter.model.feature_extractor(adapter._tensor(prepared.rgb_float32))
        if kind == 'fine':
            target = tensors['layer1'].shape[-2:]
            features = torch.cat([tensors['layer1'], F.interpolate(tensors['layer2'], size=target, mode='nearest'),
                                  F.interpolate(tensors['layer3'], size=target, mode='nearest')], 1)
            features = F.conv2d(features, projection)
        else:
            f2 = tensors['layer2']
            features = torch.cat([f2, F.interpolate(tensors['layer3'], size=f2.shape[-2:], mode='nearest')], 1)
        features = F.avg_pool2d(features, 3, stride=1, padding=1)[0]
        arr = features.numpy().astype(np.float16)
        rid = digest(row['id'])
        np.save(directory/f'{rid}.npy', arr, allow_pickle=False)
        roi = F.interpolate(torch.from_numpy(prepared.valid_roi.astype(np.float32))[None, None], size=arr.shape[-2:], mode='nearest')[0, 0].bool().numpy()
        np.save(directory/f'{rid}-roi.npy', roi, allow_pickle=False)
        index[row['id']] = {'key': rid, 'role': row['v2_role'], 'transform': transforms}
        if row['v2_role'] == 'positive_train':
            mask = image_array(row, m['config'], mask=True)
            target = F.adaptive_max_pool2d(torch.from_numpy(mask.astype(np.float32))[None, None], arr.shape[-2:])[0, 0].numpy()
            if not target.any():
                raise ValueError('A training defect target disappeared')
            np.save(directory/f'{rid}-target.npy', target, allow_pickle=False)
        if i % 100 == 0:
            print(kind, 'cache', i, '/', len(selected), flush=True)
    write_json(directory/'index.json', {'shape': list(arr.shape), 'rows': index, 'manifest_identity': m['identity'],
               'description': 'fine 64x160 multilevel features with fixed 448-to-96 random channel projection' if kind == 'fine' else 'one fixed mild image-photometric variant per training original'})


def focal_loss(logits, target, positive_weight=8., gamma=2.):
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(positive_weight), reduction='none')
    probability = logits.sigmoid()
    pt = probability*target+(1-probability)*(1-target)
    return loss*(1-pt).pow(gamma)


def train(root, variant, *, seed=None, strategy_override=None):
    parent, m, cfg = context(root)
    seed = cfg['seed'] if seed is None else seed
    fine = variant.startswith('fine')
    use_focal = variant.startswith('focal') or variant.startswith('combined')
    lighting = variant.startswith('lighting') or variant.startswith('combined')
    base = root/'fine_features' if fine else parent/'features'
    meta = cache_index(base)
    index = meta['rows']
    rows = training_rows(m)
    torch.manual_seed(seed)
    channels, h, w = meta['shape']
    sums, squares = torch.zeros(channels), torch.zeros(channels)
    count = 0
    for row in rows:
        f = torch.from_numpy(cached(base, index[row['id']]))
        sums += f.sum((1, 2))
        squares += f.square().sum((1, 2))
        count += h*w
    mean = (sums/count).reshape(1, -1, 1, 1)
    std = (squares/count-(sums/count).square()).clamp_min(.01).sqrt().reshape(1, -1, 1, 1)
    roi = torch.from_numpy(np.stack([roi_array(base, index[r['id']]) for r in rows]))
    target = torch.from_numpy(np.stack([np.load(base/f"{index[r['id']]['key']}-target.npy", allow_pickle=False)
                             if r['label'] else np.zeros((h, w), np.float32) for r in rows]))
    y = torch.tensor([float(r['label']) for r in rows])
    head_cfg = m['config']['supervised']
    head = DefectHead(channels, head_cfg['hidden_channels'])
    optimizer = torch.optim.AdamW(head.parameters(), lr=head_cfg['learning_rate'], weight_decay=head_cfg['weight_decay'])
    generator = torch.Generator().manual_seed(seed)
    aug_meta = cache_index(root/'lighting_features')['rows'] if lighting else None
    history = []
    start = time.perf_counter()
    for epoch in range(head_cfg['epochs']):
        head.train()
        order = torch.randperm(len(rows), generator=generator)
        total = 0.
        for offset in range(0, len(rows), head_cfg['batch_size']):
            ids = order[offset:offset+head_cfg['batch_size']]
            batch = []
            for i in ids:
                row = rows[int(i)]
                augment = lighting and float(torch.rand((), generator=generator)) < cfg['lighting']['training_probability']
                batch.append(cached(root/'lighting_features', aug_meta[row['id']]) if augment else cached(base, index[row['id']]))
            x = torch.from_numpy(np.stack(batch))
            optimizer.zero_grad(set_to_none=True)
            pixel, image = head((x-mean)/std, roi[ids])
            if use_focal:
                pixel_loss = focal_loss(pixel, target[ids], cfg['focal']['positive_weight'], cfg['focal']['gamma'])
            else:
                pixel_loss = F.binary_cross_entropy_with_logits(pixel, target[ids], pos_weight=torch.tensor(head_cfg['pixel_positive_weight']), reduction='none')
            loss = (pixel_loss*roi[ids]).sum()/roi[ids].sum()+F.binary_cross_entropy_with_logits(image, y[ids])
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach())*len(ids)
        history.append({'epoch': epoch+1, 'training_loss': total/len(rows)})
        if epoch % 5 == 0:
            print(variant, 'epoch', epoch+1, 'loss', history[-1]['training_loss'], flush=True)
    fit_seconds = time.perf_counter()-start
    del optimizer, target
    gc.collect()
    output = root/variant
    output.mkdir(exist_ok=False)
    torch.save({'head': head.state_dict(), 'mean': mean, 'std': std}, output/'head.pt')
    write_json(output/'model.json', {'channels': channels, 'hidden': head_cfg['hidden_channels'], 'seed': seed,
                                    'cache': str(base), 'fine': fine, 'focal': use_focal, 'lighting': lighting})
    restored, state, _ = model_load(output)
    scores, cal = {}, {}
    with torch.inference_mode():
        for role in ['normal_calibration', 'normal_evaluation', 'positive_evaluation']:
            for row in rows_for(m, role):
                entry = index[row['id']]
                f = torch.from_numpy(cached(base, entry))[None]
                valid = torch.from_numpy(roi_array(base, entry))[None]
                _, score = restored((f-state['mean'])/state['std'], valid)
                (cal if role == 'normal_calibration' else scores)[row['id']] = float(score[0])
    result(root, variant, m, scores, cal, details={'strategy': strategy_override or ('R04' if fine else ('R05+R08' if use_focal and lighting else ('R05' if use_focal else 'R08'))),
           'seed': seed, 'epochs': head_cfg['epochs'], 'fitting_ids': [r['id'] for r in rows], 'fit_seconds': fit_seconds,
           'training_history': history, 'model_sha256': sha256(output/'head.pt'),
           'fine_feature_cache': fine, 'focal': use_focal, 'lighting': lighting,
           'evaluation_used_for_early_stopping': False})


def padim50(root, weights):
    from component_inspector.vision.spike_worker import pretrained_adapter
    parent, m, _ = context(root)
    adapter = pretrained_adapter('ksdd2_surface', m['config'], weights)
    adapter.model.idx = adapter.model.idx[:50].clone()
    adapter.model.n_features = 50
    embeddings = np.load(parent/'baseline/fit_embeddings.npy', mmap_mode='r', allow_pickle=False)
    mean, inverse = gaussian_by_position(embeddings[:, :50])
    del embeddings
    adapter.model.gaussian.mean = mean
    adapter.model.gaussian.inv_covariance = inverse
    adapter.fitted, adapter.fit_count = True, 128
    output = root/'padim50'
    output.mkdir(exist_ok=False)
    torch.save(adapter.model.state_dict(), output/'state.pt')
    score, cal = {}, {}
    for role in ['normal_calibration', 'normal_evaluation', 'positive_evaluation']:
        for i, row in enumerate(rows_for(m, role)):
            p = image_array(row, m['config'])
            value, _ = adapter.predict_array(p.rgb_float32, p.valid_roi)
            (cal if role == 'normal_calibration' else score)[row['id']] = value
            if i % 100 == 0:
                print('padim50', role, i, flush=True)
    result(root, 'padim50', m, score, cal, details={'strategy': 'R09', 'retained_channels': adapter.model.idx.tolist(),
           'fit_ids': m['original_fit_ids'], 'covariance_rule': 'same pinned covariance and ridge; nested 50 of parent 100 channels',
           'state_sha256': sha256(output/'state.pt')})


def cdf_transform(reference, values):
    reference = np.sort(np.asarray(reference, np.float64))
    return np.searchsorted(reference, np.asarray(values, np.float64), side='right')/(len(reference)+1.)


@torch.inference_mode()
def stress(root, weights):
    """Paired observed-development transforms, fixed original model cutoffs."""
    from component_inspector.vision.spike_worker import pretrained_adapter
    from component_inspector.vision.geometry import prepare
    parent, m, _ = context(root)
    adapter = pretrained_adapter('ksdd2_surface', m['config'], weights)
    models = {}
    for name, path in [('supervised', parent/'supervised'), ('lighting', root/'lighting')]:
        if not (path/'result.json').exists():
            continue
        head, state, _ = model_load(path)
        cutoff = read_json(path/'result.json')['threshold_record']['threshold']
        models[name] = (head, state, cutoff)
    transformations = {'brightness_0.6': lambda im: ImageEnhance.Brightness(im).enhance(.6),
                       'brightness_1.4': lambda im: ImageEnhance.Brightness(im).enhance(1.4),
                       'blur_2': lambda im: im.filter(ImageFilter.GaussianBlur(2))}
    rows = rows_for(m, 'normal_evaluation')+rows_for(m, 'positive_evaluation')
    predictions = {name: {t: [] for t in transformations} for name in models}
    for i, row in enumerate(rows):
        guard([row])
        if sha256(row['path']) != row['hashes']['sha256']:
            raise ValueError('Source changed during stress evaluation')
        with Image.open(row['path']) as image:
            image = image.convert('RGB')
            for transform, apply in transformations.items():
                prepared = prepare(apply(image), m['config'])
                tensors = adapter.model.feature_extractor(adapter._tensor(prepared.rgb_float32))
                f2, f3 = tensors['layer2'], tensors['layer3']
                embedding = torch.cat([f2, F.interpolate(f3, size=f2.shape[-2:], mode='nearest')], 1)
                embedding = F.avg_pool2d(embedding, 3, stride=1, padding=1).half().float()
                roi = F.interpolate(torch.from_numpy(prepared.valid_roi.astype(np.float32))[None, None], size=embedding.shape[-2:], mode='nearest')[0, 0].bool()[None]
                for name, (head, state, cutoff) in models.items():
                    _, score = head((embedding-state['mean'])/state['std'], roi)
                    value = float(score[0])
                    predictions[name][transform].append({'id': row['id'], 'label': row['label'], 'status': 'scored',
                                                         'score': value, 'flagged': value > cutoff})
        if i % 100 == 0:
            print('paired stress', i, '/', len(rows), flush=True)
    truth = {r['id']: r['label'] for r in rows}
    value = {'strategy': 'R08', 'claim': 'paired stress on already observed development items; not independent validation',
             'models': {name: {t: {'metrics': evaluate(preds, truth), 'predictions': preds}
                               for t, preds in transforms.items()} for name, transforms in predictions.items()}}
    write_json(root/'stress/result.json', value)
    for name, transforms in value['models'].items():
        for t, d in transforms.items():
            print('STRESS', name, t, d['metrics']['recall']['rate'], d['metrics']['accepted_normal_fpr']['rate'], flush=True)


def ensemble(root):
    parent, m, cfg = context(root)
    candidates = [('supervised', parent/'supervised/result.json')]
    for name in ['fine', 'focal', 'lighting', 'combined']:
        p = root/name/'result.json'
        if p.exists():
            candidates.append((name, p))
    eligible = [(n, read_json(p)) for n, p in candidates]
    eligible = [(n, d) for n, d in eligible if d['metrics']['accepted_normal_fpr']['rate'] < .10]
    name, first = max(eligible, key=lambda v: (v[1]['metrics']['recall']['rate'], -v[1]['metrics']['accepted_normal_fpr']['rate']))
    second = read_json(parent/'patchcore/result.json')
    parents = [first, second]
    cal_ids = sorted([r['id'] for r in rows_for(m, 'normal_calibration')], key=lambda i: digest([cfg['seed'], 'ensemble_normalization', i]))
    cut = round(len(cal_ids)*cfg['ensemble']['normalization_fraction'])
    norm_ids, threshold_ids = cal_ids[:cut], cal_ids[cut:]
    assert not set(norm_ids)&set(threshold_ids)
    eval_ids = sorted(r['id'] for r in first['predictions'])
    normalizers, zvalues, cdfs = [], [], []
    all_ids = threshold_ids+eval_ids
    for source in parents:
        cal = {r['id']: r['score'] for r in source['calibration_scores']}
        prediction = {r['id']: r['score'] for r in source['predictions']}
        normal = np.asarray([cal[i] for i in norm_ids])
        values = np.asarray([cal[i] if i in cal else prediction[i] for i in all_ids])
        median = float(np.median(normal))
        iqr = max(float(np.percentile(normal, 75)-np.percentile(normal, 25)), 1e-6)
        zvalues.append((values-median)/iqr)
        cdfs.append(cdf_transform(normal, values))
        normalizers.append({'median': median, 'iqr': iqr})
    combined = {'maximum_normal_cdf': np.maximum(cdfs[0], cdfs[1]),
                'head_0.75_patchcore_0.25_robust_z': .75*zvalues[0]+.25*zvalues[1]}
    for rule, values in combined.items():
        result(root, 'ensemble_'+rule, m, dict(zip(eval_ids, values[len(threshold_ids):])),
               dict(zip(threshold_ids, values[:len(threshold_ids)])),
               details={'strategy': 'R10', 'parents': [name, 'round1/patchcore'],
                        'normalization_ids': norm_ids, 'threshold_ids': threshold_ids,
                        'normalizers': normalizers, 'fixed_rule': rule,
                        'head_parent_selection': cfg['selection'], 'selection_exposure': 'observed development results'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['prepare', 'aggregation', 'fine_cache', 'lighting_cache', 'fine', 'focal', 'lighting', 'combined', 'padim50', 'ensemble', 'stress'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path, default=PROJECT/'outputs/recall-round1')
    parser.add_argument('--weights', type=Path, default=PROJECT.parent/'.cache/recall-downloads/model.safetensors')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.phase == 'prepare':
        cfg_path = PROJECT/'configs/recall_round2.json'
        m = load_manifest(args.parent.resolve())
        write_json(root/'protocol.json', {'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
                   'parent': str(args.parent.resolve()), 'manifest_identity': m['identity'],
                   'config': read_json(cfg_path), 'config_sha256': sha256(cfg_path),
                   'source_sha256': sha256(Path(__file__)),
                   'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()})
        return
    if (root/f'{args.phase}-complete.json').exists():
        raise ValueError('Phase already complete; inspect evidence instead of rerunning')
    start = time.perf_counter()
    if args.phase == 'aggregation':
        aggregation(root)
    elif args.phase.endswith('_cache'):
        make_cache(root, args.phase.split('_')[0], args.weights)
    elif args.phase == 'padim50':
        padim50(root, args.weights)
    elif args.phase == 'ensemble':
        ensemble(root)
    elif args.phase == 'stress':
        stress(root, args.weights)
    else:
        train(root, args.phase, seed=args.seed)
    write_json(root/f'{args.phase}-complete.json', {'elapsed_seconds': time.perf_counter()-start,
               'source_sha256': sha256(Path(__file__)), 'phase': args.phase})


if __name__ == '__main__':
    main()
