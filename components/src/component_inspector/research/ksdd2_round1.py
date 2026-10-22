"""Bounded CPU KSDD2 development comparison: P0, supervised, PatchCore, coverage.

Official test rows are rejected before image decoding. Historical files are read
only. Every phase writes an exclusive completion file and immutable predictions.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from component_inspector.data.acquire import sha256
from component_inspector.evaluation.calibration import calibrate
from component_inspector.evaluation.metrics import evaluate
from component_inspector.vision.geometry import prepare


RECIPE = 'ksdd2_surface'
PROJECT = Path(__file__).resolve().parents[3]
HISTORICAL = PROJECT/'data/manifests/stage2/b769d8598112008728d354145526815eceeaaf32e477861af22c25a541a1ecc9/manifest.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def guard(records):
    """Fail before decode if a historical test/unknown/duplicate group leaks in."""
    groups = {}
    for row in records:
        if (row['recipe'] != RECIPE or row['official_split'] != 'train'
                or row.get('quarantine_reason') or row['label'] not in (0, 1)):
            raise ValueError('Only audited nonquarantined KSDD2 training originals are allowed')
        role = row['v2_role']
        if role in {'normal_fit', 'normal_calibration', 'normal_evaluation'} and row['label'] != 0:
            raise ValueError('Defect in a normal-only role')
        if role in {'positive_train', 'positive_evaluation'} and row['label'] != 1:
            raise ValueError('Normal in a positive role')
        if role not in {'normal_fit', 'normal_calibration', 'normal_evaluation', 'positive_train', 'positive_evaluation'}:
            raise ValueError('Unknown v2 role')
        group = row['group']
        if group in groups and groups[group] != role:
            raise ValueError('Original/duplicate group crosses roles')
        groups[group] = role


def build_manifest(raw_root, config):
    old = read_json(HISTORICAL)
    records = []
    eligible = [r for r in old['splits']['records'] if r['recipe'] == RECIPE
                and r['official_split'] == 'train' and not r.get('quarantine_reason')]
    positive_groups = sorted({r['group'] for r in eligible if r['label'] == 1},
                             key=lambda g: digest([config['seed'], 'supervised_positive', g]))
    train_groups = set(positive_groups[:round(len(positive_groups)*config['positive_train_fraction'])])
    for original in eligible:
        row = dict(original)
        if row['label'] == 1:
            role = 'positive_train' if row['group'] in train_groups else 'positive_evaluation'
        else:
            role = {'fit': 'normal_fit', 'unused_fit': 'normal_fit',
                    'development': 'normal_evaluation', 'calibration': 'normal_calibration'}[row['experiment_role']]
        relative = row['provenance']['source_relative_path']
        path = (Path(raw_root)/relative).resolve()
        if not path.is_relative_to(Path(raw_root).resolve()):
            raise ValueError('Source path escapes raw root')
        row.update(path=str(path), mask_path=str(path.with_name(path.stem+'_GT.png')),
                   v2_role=role, historical_path=original['path'])
        records.append(row)
    guard(records)
    original_fit = [i for i in old['splits']['fit_ids'] if i.startswith('ksdd2:')]
    assert len(original_fit) == config['normal_fit_count']
    result = {'schema_version': 2, 'recipe': RECIPE, 'claim': config['evaluation_claim'],
              'historical_manifest_canonical_hash': digest(old),
              'config': config, 'records': records, 'original_fit_ids': original_fit,
              'counts': {role: sum(r['v2_role'] == role for r in records) for role in sorted({r['v2_role'] for r in records})},
              'physical_item_independence': 'unknown; inherited appearance groups only'}
    result['identity'] = digest(result)
    return result


def load_manifest(root):
    m = read_json(root/'manifest.json')
    expected = m.pop('identity')
    if digest(m) != expected:
        raise ValueError('v2 manifest changed')
    m['identity'] = expected
    guard(m['records'])
    return m


def image_array(row, config, *, mask=False):
    guard([row])
    path = row['mask_path'] if mask else row['path']
    expected = row['hashes']['mask_sha256' if mask else 'sha256']
    if sha256(path) != expected:
        raise ValueError('Source image/mask hash mismatch')
    with Image.open(path) as im:
        if not mask:
            return prepare(im, config)
        # Same rotate/letterbox geometry, but labels use nearest-neighbor sampling.
        im = im.convert('L').transpose(Image.Transpose.ROTATE_270)
        h, w = config['input']['height'], config['input']['width']
        scale = min(w/im.width, h/im.height)
        size = max(1, round(im.width*scale)), max(1, round(im.height*scale))
        target = np.zeros((h, w), np.uint8)
        left, top = (w-size[0])//2, (h-size[1])//2
        target[top:top+size[1], left:left+size[0]] = np.asarray(im.resize(size, Image.Resampling.NEAREST)) > 0
        return target


def rows_for(m, role):
    return sorted((r for r in m['records'] if r['v2_role'] == role), key=lambda r: r['id'])


def scored_report(root, name, m, scores, calibration_scores, resources, extra=None):
    evaluation_rows = rows_for(m, 'normal_evaluation')+rows_for(m, 'positive_evaluation')
    expected_ids = {r['id'] for r in evaluation_rows}
    if set(scores) != expected_ids or len(calibration_scores) != len(rows_for(m, 'normal_calibration')):
        raise ValueError('Incomplete calibration/evaluation predictions')
    threshold = calibrate([x['score'] for x in calibration_scores], alpha=m['config']['normal_calibration_alpha'])
    predictions = [{**scores[r['id']], 'id': r['id'], 'label': r['label'], 'group': r['group'],
                    'status': 'scored', 'flagged': bool(scores[r['id']]['score'] > threshold['threshold'])}
                   for r in evaluation_rows]
    truth = {r['id']: r['label'] for r in evaluation_rows}
    result = {'method': name, 'claim': m['claim'], 'manifest_identity': m['identity'],
              'threshold_record': threshold, 'metrics': evaluate(predictions, truth),
              'predictions': predictions, 'calibration_scores': calibration_scores,
              'resources': resources, **(extra or {})}
    write_json(root/f'{name}/result.json', result)
    print(json.dumps({'method': name, 'counts': result['metrics']['counts'],
                      'recall': result['metrics']['recall']['rate'],
                      'fpr': result['metrics']['accepted_normal_fpr']['rate']}), flush=True)
    return result


def gaussian_by_position(embeddings, chunk=128):
    """Pinned PaDiM equations in position chunks; no full covariance temporary.

    Embeddings may be a disk memmap [samples, channels, positions]. Keep each
    sample's complete fit ordering and the library's exact covariance function.
    """
    from anomalib.models.components.stats.multi_variate_gaussian import MultiVariateGaussian
    n, channels, positions = embeddings.shape
    mean = torch.empty(channels, positions)
    inverse = torch.empty(positions, channels, channels)
    identity = torch.eye(channels)
    for start in range(0, positions, chunk):
        end = min(start+chunk, positions)
        block = torch.from_numpy(np.array(embeddings[:, :, start:end], dtype=np.float32))
        mean[:, start:end] = block.mean(0)
        cov = torch.stack([MultiVariateGaussian._cov(block[:, :, j], rowvar=False)+.01*identity
                           for j in range(end-start)])
        inverse[start:end] = torch.linalg.inv(cov+1e-5*identity)
    return mean, inverse


def baseline(root, m, weights):
    from component_inspector.vision.spike_worker import pretrained_adapter
    from component_inspector.vision.padim_adapter import PadimAdapter
    cfg = m['config']
    ids = set(m['original_fit_ids'])
    fit = sorted((r for r in m['records'] if r['id'] in ids), key=lambda r: r['id'])
    # Historical request/manifest ordering was sorted by ID; retain it exactly.
    adapter = pretrained_adapter(RECIPE, cfg, weights)
    start = time.perf_counter()
    # This laptop has less free memory than the teammate's run. Preserve the
    # pinned model/equations but write fit embeddings to disk and invert one
    # group of positions at a time instead of materializing three 400-MB arrays.
    work = root/'baseline'
    work.mkdir(exist_ok=False)
    with torch.inference_mode():
        store = None
        for i, row in enumerate(fit):
            p = image_array(row, cfg)
            tensors = adapter.model.feature_extractor(adapter._tensor(p.rgb_float32))
            embedding = adapter.model.generate_embedding(tensors)[0].reshape(100, -1).cpu().numpy()
            if store is None:
                store = np.lib.format.open_memmap(work/'fit_embeddings.npy', mode='w+', dtype=np.float32,
                                                 shape=(len(fit), *embedding.shape))
            store[i] = embedding
        store.flush()
        del store
        store = np.load(work/'fit_embeddings.npy', mmap_mode='r', allow_pickle=False)
        mean, inverse = gaussian_by_position(store)
        del store
        adapter.model.gaussian.mean = mean
        adapter.model.gaussian.inv_covariance = inverse
        adapter.fitted = True
        adapter.fit_count = len(fit)
    fit_seconds = time.perf_counter()-start
    target = root/'baseline/model'
    adapter.save(target, {'fit_manifest_hash': m['identity'], 'fit_ids': [r['id'] for r in fit]})
    sample = image_array(fit[0], cfg)
    before, before_map = adapter.predict_array(sample.rgb_float32, sample.valid_roi)
    del adapter
    gc.collect()
    adapter = PadimAdapter.load(target, expected_recipe=RECIPE)
    after, after_map = adapter.predict_array(sample.rgb_float32, sample.valid_roi)
    if not np.allclose(before_map, after_map, atol=1e-5, rtol=1e-5):
        raise ValueError('Baseline save/load parity failed')
    scores, calibration_scores, timing = {}, [], []
    for role in ['normal_calibration', 'normal_evaluation', 'positive_evaluation']:
        for i, row in enumerate(rows_for(m, role)):
            tick = time.perf_counter()
            p = image_array(row, cfg)
            score, _ = adapter.predict_array(p.rgb_float32, p.valid_roi)
            elapsed = time.perf_counter()-tick
            timing.append(elapsed)
            value = {'id': row['id'], 'score': score, 'seconds': elapsed}
            if role == 'normal_calibration':
                calibration_scores.append(value)
            else:
                scores[row['id']] = value
            if i % 100 == 0:
                print('baseline', role, i, flush=True)
    historical = read_json(PROJECT/'historical_benchmark/frozen-20261019T041652Z-9c825801/calibration/ksdd2_surface_padim/ksdd2_surface_padim.json')
    old_scores = {r['id']: r['score'] for r in historical['rows']}
    delta = max(abs(r['score']-old_scores[r['id']]) for r in calibration_scores)
    return scored_report(root, 'baseline', m, scores, calibration_scores,
                         {'fit_seconds': fit_seconds, 'warm_p95_seconds': float(np.percentile(timing[1:], 95))},
                         {'historical_calibration_max_abs_score_delta': delta,
                          'historical_calibration_allclose_1e5': bool(all(np.isclose(r['score'], old_scores[r['id']], atol=1e-5, rtol=1e-5) for r in calibration_scores)),
                          'save_load_score_delta': abs(before-after), 'fit_ids': [r['id'] for r in fit],
                          'fit_execution': 'disk-backed embeddings; pinned covariance equations in 128-position chunks; historical calibration parity checked'})


@torch.inference_mode()
def features(root, m, weights):
    from component_inspector.vision.spike_worker import pretrained_adapter
    adapter = pretrained_adapter(RECIPE, m['config'], weights)
    directory = root/'features'
    directory.mkdir(exist_ok=False)
    mapping = {}
    original = set(m['original_fit_ids'])
    # R03 needs pooled embeddings for the entire normal fit pool. Cache all
    # permitted full maps to disk, never keep the pool resident in memory.
    for i, row in enumerate(sorted(m['records'], key=lambda r: r['id'])):
        p = image_array(row, m['config'])
        start = time.perf_counter()
        tensors = adapter.model.feature_extractor(adapter._tensor(p.rgb_float32))
        f2, f3 = tensors['layer2'], tensors['layer3']
        embedding = torch.cat([f2, F.interpolate(f3, size=f2.shape[-2:], mode='nearest')], dim=1)
        embedding = F.avg_pool2d(embedding, 3, stride=1, padding=1)[0]
        arr = embedding.cpu().numpy().astype(np.float16)
        rid = digest(row['id'])
        np.save(directory/f'{rid}.npy', arr, allow_pickle=False)
        roi = F.interpolate(torch.from_numpy(p.valid_roi.astype(np.float32))[None, None], size=arr.shape[-2:], mode='nearest')[0, 0].bool().numpy()
        np.save(directory/f'{rid}-roi.npy', roi, allow_pickle=False)
        mapping[row['id']] = {'key': rid, 'role': row['v2_role'], 'original_fit': row['id'] in original,
                              'global_embedding': arr.astype(np.float32)[:, roi].mean(axis=1).tolist(),
                              'extract_seconds': time.perf_counter()-start}
        if row['v2_role'] == 'positive_train':
            label = image_array(row, m['config'], mask=True)
            target = F.adaptive_max_pool2d(torch.from_numpy(label.astype(np.float32))[None, None], arr.shape[-2:])[0, 0].numpy()
            np.save(directory/f'{rid}-target.npy', target, allow_pickle=False)
        if i % 100 == 0:
            print('features', i, '/', len(m['records']), flush=True)
    write_json(directory/'index.json', {'manifest_identity': m['identity'], 'shape': list(arr.shape), 'rows': mapping})


def farthest_points(values, count, *, seed=4217):
    """Deterministic greedy normal-feature coverage; no defect labels used."""
    x = torch.as_tensor(values, dtype=torch.float32)
    if not 1 <= count <= len(x):
        raise ValueError('Invalid diversity selection count')
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite diversity features')
    center = x.mean(0)
    selected = [int(((x-center)**2).sum(1).argmax())]
    nearest = torch.full((len(x),), float('inf'))
    used = torch.zeros(len(x), dtype=torch.bool)
    for _ in range(count-1):
        distance = ((x-x[selected[-1]])**2).sum(1)
        nearest = torch.minimum(nearest, distance)
        used[selected[-1]] = True
        nearest[used] = -1
        selected.append(int(nearest.argmax()))
    return selected


def load_feature(root, info):
    return np.load(root/'features'/f"{info['key']}.npy", allow_pickle=False).astype(np.float32)


def build_bank(root, index, fit_ids, config):
    rng = np.random.default_rng(config['seed'])
    pc = config['patchcore']
    per_image = max(1, pc['candidate_pool_cap']//len(fit_ids))
    candidates = []
    total = 0
    for identity in fit_ids:
        info = index[identity]
        if info['role'] != 'normal_fit':
            raise ValueError('PatchCore attempted to fit outside normal fitting role')
        f = load_feature(root, info)
        roi = np.load(root/'features'/f"{info['key']}-roi.npy", allow_pickle=False)
        patches = f[:, roi].T
        total += len(patches)
        take = rng.choice(len(patches), min(per_image, len(patches)), replace=False)
        candidates.append(patches[take])
    pool = np.concatenate(candidates)
    # This explicit per-image candidate cap is a CPU approximation, recorded as
    # such; final bank is selected with greedy coverage in a fixed projection.
    projection = rng.standard_normal((pool.shape[1], pc['projection_dimensions']), dtype=np.float32)/math.sqrt(pc['projection_dimensions'])
    projected = pool@projection
    count = min(pc['bank_cap'], math.ceil(total*pc['coreset_ratio']), len(pool))
    chosen = farthest_points(projected, count, seed=config['seed'])
    return torch.from_numpy(pool[chosen].copy()), {'total_fit_patches': total, 'candidate_patches': len(pool), 'bank_vectors': count,
                                                  'candidate_sampling': 'fixed per-normal-image random cap before projected greedy coreset'}


@torch.inference_mode()
def bank_score(features, roi, bank, chunk=256):
    query = torch.from_numpy(features[:, roi].T.copy())
    if not len(query) or not torch.isfinite(query).all():
        raise ValueError('Invalid query/ROI')
    nearest = []
    for start in range(0, len(query), chunk):
        nearest.append(torch.cdist(query[start:start+chunk], bank).min(1).values)
    return float(torch.cat(nearest).max())


def patchcore(root, m, *, coverage=False):
    index = read_json(root/'features/index.json')['rows']
    if coverage:
        pool = sorted(r['id'] for r in rows_for(m, 'normal_fit'))
        globals_ = np.asarray([index[i]['global_embedding'] for i in pool], np.float32)
        selected = farthest_points(globals_, m['config']['expanded_normal_fit_count'])
        variants = [('coverage128', [pool[i] for i in selected[:128]]),
                    ('coverage256', [pool[i] for i in selected])]
    else:
        variants = [('patchcore', sorted(m['original_fit_ids']))]
    for name, fit_ids in variants:
        start = time.perf_counter()
        bank, metadata = build_bank(root, index, fit_ids, m['config'])
        fit_seconds = time.perf_counter()-start
        path = root/name
        path.mkdir(exist_ok=False)
        torch.save(bank, path/'bank.pt')
        # Tensor-only trusted artifact parity check.
        restored = torch.load(path/'bank.pt', weights_only=True)
        if not torch.equal(bank, restored):
            raise ValueError('Patch bank save/load mismatch')
        scores, calibration_scores, timing = {}, [], []
        for role in ['normal_calibration', 'normal_evaluation', 'positive_evaluation']:
            for i, row in enumerate(rows_for(m, role)):
                info = index[row['id']]
                f = load_feature(root, info)
                roi = np.load(root/'features'/f"{info['key']}-roi.npy", allow_pickle=False)
                tick = time.perf_counter()
                score = bank_score(f, roi, bank, m['config']['patchcore']['query_chunk'])
                elapsed = time.perf_counter()-tick+info['extract_seconds']
                timing.append(elapsed)
                value = {'id': row['id'], 'score': score, 'seconds': elapsed}
                if role == 'normal_calibration':
                    calibration_scores.append(value)
                else:
                    scores[row['id']] = value
                if i % 100 == 0:
                    print(name, role, i, flush=True)
        scored_report(root, name, m, scores, calibration_scores,
                      {'fit_seconds': fit_seconds, 'warm_p95_seconds': float(np.percentile(timing[1:], 95)),
                       'timing_scope': 'cached scoring plus separately measured backbone extraction; excludes media decode'},
                      {'fit_ids': fit_ids, 'bank': metadata, 'bank_sha256': sha256(path/'bank.pt')})


class DefectHead(nn.Module):
    """Trainable local defect map plus learned image classifier on frozen features."""
    def __init__(self, channels=384, hidden=32):
        super().__init__()
        self.segment = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, 1, 1))
        self.classify = nn.Linear(3, 1)

    def forward(self, x, roi):
        values = self.segment(x)[:, 0]
        masked = values.masked_fill(~roi, -1e4)
        flat = masked.flatten(1)
        k = max(1, math.ceil(flat.shape[1]*.005))
        top = flat.topk(k, dim=1).values.mean(1)
        maximum = flat.max(1).values
        mean = (values*roi).flatten(1).sum(1)/roi.flatten(1).sum(1).clamp_min(1)
        image = self.classify(torch.stack([maximum, top, mean], dim=1))[:, 0]
        return values, image


def supervised(root, m):
    torch.manual_seed(m['config']['seed'])
    index = read_json(root/'features/index.json')['rows']
    training_ids = sorted(m['original_fit_ids'])+sorted(r['id'] for r in rows_for(m, 'positive_train'))
    info = [index[i] for i in training_ids]
    shape = read_json(root/'features/index.json')['shape']
    # Training-only channel normalization, persisted for inference.
    sums, squares = torch.zeros(shape[0]), torch.zeros(shape[0])
    count = 0
    for entry in info:
        sample = torch.from_numpy(load_feature(root, entry))
        sums += sample.sum((1, 2))
        squares += sample.square().sum((1, 2))
        count += sample.shape[1]*sample.shape[2]
    mean = (sums/count).reshape(1, -1, 1, 1)
    std = (squares/count-(sums/count).square()).clamp_min(.01).sqrt().reshape(1, -1, 1, 1)
    roi = torch.from_numpy(np.stack([np.load(root/'features'/f"{r['key']}-roi.npy", allow_pickle=False) for r in info]))
    labels = torch.tensor([float(index[i]['role'] == 'positive_train') for i in training_ids])
    targets = torch.from_numpy(np.stack([np.load(root/'features'/f"{r['key']}-target.npy", allow_pickle=False)
                                        if index[i]['role'] == 'positive_train' else np.zeros(shape[-2:], np.float32)
                                        for i, r in zip(training_ids, info)]))
    cfg = m['config']['supervised']
    head = DefectHead(shape[0], cfg['hidden_channels'])
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    generator = torch.Generator().manual_seed(m['config']['seed'])
    history = []
    start = time.perf_counter()
    for epoch in range(cfg['epochs']):
        head.train()
        order = torch.randperm(len(info), generator=generator)
        total = 0.
        for offset in range(0, len(info), cfg['batch_size']):
            ids = order[offset:offset+cfg['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            batch = torch.from_numpy(np.stack([load_feature(root, info[int(i)]) for i in ids]))
            pixel, image = head((batch-mean)/std, roi[ids])
            loss_map = F.binary_cross_entropy_with_logits(pixel, targets[ids], pos_weight=torch.tensor(cfg['pixel_positive_weight']), reduction='none')
            loss_seg = (loss_map*roi[ids]).sum()/roi[ids].sum()
            loss = loss_seg+cfg['image_loss_weight']*F.binary_cross_entropy_with_logits(image, labels[ids])
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach())*len(ids)
        history.append({'epoch': epoch+1, 'training_loss': total/len(info)})
        if epoch % 5 == 0:
            print('supervised epoch', epoch+1, 'loss', history[-1]['training_loss'], flush=True)
    fit_seconds = time.perf_counter()-start
    del targets, optimizer
    gc.collect()
    head.eval()
    output = root/'supervised'
    output.mkdir(exist_ok=False)
    torch.save({'head': head.state_dict(), 'mean': mean, 'std': std}, output/'head.pt')
    restored = torch.load(output/'head.pt', weights_only=True)
    test_head = DefectHead(384, cfg['hidden_channels'])
    test_head.load_state_dict(restored['head'])
    if any(not torch.equal(v, restored['head'][k]) for k, v in head.state_dict().items()):
        raise ValueError('Supervised artifact save/load mismatch')
    scores, calibration_scores, timing = {}, [], []
    with torch.inference_mode():
        for role in ['normal_calibration', 'normal_evaluation', 'positive_evaluation']:
            for row in rows_for(m, role):
                r = index[row['id']]
                f = torch.from_numpy(load_feature(root, r))[None]
                valid = torch.from_numpy(np.load(root/'features'/f"{r['key']}-roi.npy", allow_pickle=False))[None]
                tick = time.perf_counter()
                _, prediction = head((f-mean)/std, valid)
                elapsed = time.perf_counter()-tick+r['extract_seconds']
                timing.append(elapsed)
                value = {'id': row['id'], 'score': float(prediction[0]), 'seconds': elapsed}
                if role == 'normal_calibration':
                    calibration_scores.append(value)
                else:
                    scores[row['id']] = value
    scored_report(root, 'supervised', m, scores, calibration_scores,
                  {'fit_seconds': fit_seconds, 'warm_p95_seconds': float(np.percentile(timing[1:], 95)),
                   'timing_scope': 'cached head scoring plus separately measured backbone extraction; excludes media decode'},
                  {'training_ids': training_ids, 'epochs': cfg['epochs'], 'training_history': history,
                   'head_sha256': sha256(output/'head.pt'), 'mask_targets': 'nearest geometry; adaptive max pooling to retain thin defect targets',
                   'architecture': 'frozen ResNet18 layers2/3; 1x1 local head; image classifier on max/top0.5pct/mean logits; CPU adaptation, not official SuperSimpleNet'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare', 'baseline', 'features', 'supervised', 'patchcore', 'coverage'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--config', type=Path, default=PROJECT/'configs/recall_round1.json')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(4217)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.phase == 'prepare':
        m = build_manifest(args.raw_root, read_json(args.config))
        write_json(root/'manifest.json', m)
        write_json(root/'environment.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
                   'versions': {n: version(n) for n in ['torch', 'torchvision', 'anomalib', 'timm', 'numpy', 'Pillow', 'scipy']},
                   'config_sha256': sha256(args.config), 'code_sha256': sha256(Path(__file__))})
        print(m['counts'], flush=True)
        return
    if (root/f'{args.phase}-complete.json').exists():
        raise ValueError('Phase already completed; inspect saved output instead of overwriting')
    m = load_manifest(root)
    start = time.perf_counter()
    if args.phase == 'baseline':
        baseline(root, m, args.weights)
    elif args.phase == 'features':
        features(root, m, args.weights)
    elif args.phase == 'supervised':
        supervised(root, m)
    else:
        patchcore(root, m, coverage=args.phase == 'coverage')
    write_json(root/f'{args.phase}-complete.json', {'elapsed_seconds': time.perf_counter()-start,
               'manifest_identity': m['identity'], 'code_sha256': sha256(Path(__file__))})


if __name__ == '__main__':
    main()
