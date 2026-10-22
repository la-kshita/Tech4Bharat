"""Training-only nested CV; official test inputs are ineligible in every phase."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import math
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, grey_opening, grey_closing
from scipy.stats import binomtest
import torch
from torch import nn
from torch.nn import functional as F

from component_inspector.data.acquire import sha256
from component_inspector.evaluation.calibration import calibrate
from component_inspector.evaluation.metrics import evaluate
from component_inspector.research.ksdd2_round1 import (
    digest, farthest_points, image_array, load_manifest, read_json, write_json,
)
from component_inspector.vision.spike_worker import WEIGHT_HASH


PROJECT = Path(__file__).resolve().parents[3]
CONFIG = PROJECT/'configs/recall_small_defects.json'


def guard(rows, *, allowed_ids=None, forbidden_ids=()):
    seen = set()
    for row in rows:
        if (row['recipe'] != 'ksdd2_surface' or row['official_split'] != 'train'
                or row.get('quarantine_reason') or row['label'] not in (0, 1)
                or row['id'] in seen or row['id'] in forbidden_ids
                or (allowed_ids is not None and row['id'] not in allowed_ids)):
            raise ValueError('Only permitted, nonduplicate official-training originals allowed')
        seen.add(row['id'])


def dependencies():
    from component_inspector.research import ksdd2_round1
    from component_inspector.vision import geometry, spike_worker, padim_adapter
    from component_inspector.evaluation import calibration, metrics
    from component_inspector.data import acquire
    return [Path(__file__), CONFIG, PROJECT/'tools/run_small_defects.py', *[Path(m.__file__) for m in
            (ksdd2_round1, geometry, spike_worker, padim_adapter, calibration, metrics, acquire)]]


def freeze(root, parent, weights):
    root.mkdir(parents=True, exist_ok=False)
    m, cfg = load_manifest(parent), read_json(CONFIG)
    guard(m['records'])
    reserved = sorted(r['id'] for r in m['records'] if r['v2_role'] == 'normal_calibration')
    pool = [r for r in m['records'] if r['id'] not in reserved]
    guard(pool, forbidden_ids=set(reserved))
    if len(pool) != 1914 or len(reserved) != 417 or sha256(weights) != WEIGHT_HASH:
        raise ValueError('Unexpected training population or backbone')
    groups, strata, areas = {}, {}, {}
    for row in pool:
        groups.setdefault(row['group'], []).append(row)
        if row['label']:
            target = image_array(row, m['config'], mask=True)
            # Original source coordinates define the small-defect stratum.
            with Image.open(row['mask_path']) as im:
                area = float((np.asarray(im.convert('L')) > 0).mean())
            if not target.any():
                raise ValueError('Positive original lost in geometry')
            areas[row['id']] = area
    for group, rows in groups.items():
        if len({r['label'] for r in rows}) != 1:
            raise ValueError('Conflicting labels in a group')
        label = rows[0]['label']
        small = bool(label and min(areas[r['id']] for r in rows) <= cfg['small_defect_max_area_fraction'])
        strata.setdefault((label, small), []).append(group)
    assignment = {}
    for stratum, group_ids in sorted(strata.items()):
        ordered = sorted(group_ids, key=lambda g: digest([cfg['seed'], list(stratum), g]))
        for i, group in enumerate(ordered):
            for row in groups[group]:
                assignment[row['id']] = i % cfg['folds']
    value = {'created_utc': datetime.now(timezone.utc).isoformat(),
             'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
             'source_sha256': {str(p): sha256(p) for p in dependencies()},
             'parent': str(parent), 'parent_manifest_identity': m['identity'],
             'weights': str(weights), 'weights_sha256': WEIGHT_HASH, 'config': cfg,
             'lock_sha256': sha256(PROJECT/'requirements-recall.lock'),
             'versions': {p: version(p) for p in ('torch', 'torchvision', 'anomalib', 'timm', 'numpy', 'scipy', 'Pillow')},
             'pool_ids': sorted(assignment), 'reserved_normal_calibration_ids': reserved,
             'folds': assignment, 'positive_source_area_fraction': areas,
             'counts': {str(f): dict(Counter('defective' if r['label'] else 'normal'
                        for r in pool if assignment[r['id']] == f)) for f in range(cfg['folds'])},
             'small_defects_per_fold': {str(f): sum(areas.get(r['id'], 1.) <= .005
                                      for r in pool if assignment[r['id']] == f) for f in range(cfg['folds'])},
             'official_test_policy': 'never decode, score, select on, or recalibrate using official test',
             'prior_exposure': 'all official training originals previously used or observed; repartitioning does not make pristine data',
             'physical_independence': 'unknown; keep audited appearance groups whole',
             'nested_design': 'train each inner candidate on one fold, calibrate on 200 normal groups within it; choose using the other nonouter fold; refit selected recipe on both nonouter folds; evaluate outer fold once',
             'final_selection': 'all six directed inner fold comparisons only, frozen before any outer inference; final reserved 417 normals only calibrate after model selection',
             'final_confirmation': 'unavailable without a genuinely new labeled production batch'}
    value['identity'] = digest(value)
    write_json(root/'protocol.json', value)
    print('FROZEN', value['identity'], value['counts'], value['small_defects_per_fold'], flush=True)


def context(root):
    p = read_json(root/'protocol.json')
    identity = p.pop('identity')
    if digest(p) != identity:
        raise ValueError('Protocol changed')
    p['identity'] = identity
    for file, expected in p['source_sha256'].items():
        if sha256(file) != expected:
            raise ValueError('Frozen source/config changed: '+file)
    if sha256(p['weights']) != p['weights_sha256']:
        raise ValueError('Backbone changed')
    m = load_manifest(Path(p['parent']))
    if m['identity'] != p['parent_manifest_identity']:
        raise ValueError('Parent manifest changed')
    by = {r['id']: r for r in m['records']}
    pool = [by[i] for i in p['pool_ids']]
    guard(pool, allowed_ids=set(p['pool_ids']), forbidden_ids=set(p['reserved_normal_calibration_ids']))
    return p, m, by


def morphology_score(rgb, cfg):
    gray = np.asarray(rgb.convert('L'), dtype=np.float32)/255.
    smooth = gaussian_filter(gray, cfg['gaussian_sigma'])
    response = np.zeros_like(smooth)
    for k in cfg['kernel_sizes']:
        response = np.maximum(response, smooth-grey_opening(smooth, size=(k, k)))
        response = np.maximum(response, grey_closing(smooth, size=(k, k))-smooth)
    values = response.ravel()/max(float(smooth.std()), cfg['contrast_floor'])
    k = max(1, math.ceil(.0001*len(values)))
    return float(np.partition(values, -k)[-k:].mean())


@torch.inference_mode()
def cache(root):
    from component_inspector.vision.spike_worker import pretrained_adapter
    p, m, by = context(root)
    output = root/'features'
    output.mkdir(exist_ok=False)
    adapter = pretrained_adapter('ksdd2_surface', m['config'], Path(p['weights']))
    old = Path(p['parent'])/'features'
    old_index = read_json(old/'index.json')['rows']
    index = {}
    for i, identity in enumerate(p['pool_ids']):
        row = by[identity]
        guard([row], allowed_ids=set(p['pool_ids']))
        prepared = image_array(row, m['config'])
        features = adapter.model.feature_extractor(adapter._tensor(prepared.rgb_float32))
        f1, f2, f3 = (features[f'layer{n}'] for n in (1, 2, 3))
        coarse = F.avg_pool2d(torch.cat([f2, F.interpolate(f3, size=f2.shape[-2:], mode='nearest')], 1), 3, stride=1, padding=1)[0].numpy().astype(np.float16)
        old_file = old/f"{old_index[identity]['key']}.npy"
        if not np.array_equal(coarse, np.load(old_file, allow_pickle=False)):
            raise ValueError('Native backbone output differs from original frozen feature cache')
        fine = torch.cat([f1, F.interpolate(f2, size=f1.shape[-2:], mode='nearest'),
                          F.interpolate(f3, size=f1.shape[-2:], mode='nearest')], 1)[0].numpy().astype(np.float16)
        key = digest(identity)
        feature_path, roi_path = output/f'{key}.npy', output/f'{key}-roi.npy'
        np.save(feature_path, fine, allow_pickle=False)
        roi = F.interpolate(torch.from_numpy(prepared.valid_roi.astype(np.float32))[None, None],
                            size=fine.shape[-2:], mode='nearest')[0, 0].bool().numpy()
        np.save(roi_path, roi, allow_pickle=False)
        entry = {'key': key, 'fine_path': str(feature_path), 'fine_sha256': sha256(feature_path),
                 'fine_roi_path': str(roi_path), 'fine_roi_sha256': sha256(roi_path),
                 'coarse_path': str(old_file), 'coarse_sha256': sha256(old_file),
                 'coarse_roi_path': str(old/f"{old_index[identity]['key']}-roi.npy"),
                 'coarse_roi_sha256': sha256(old/f"{old_index[identity]['key']}-roi.npy"),
                 'shape': list(fine.shape)}
        if row['label']:
            mask = image_array(row, m['config'], mask=True)
            for grid, shape in [('coarse', coarse.shape[-2:]), ('fine', fine.shape[-2:])]:
                target = F.adaptive_max_pool2d(torch.from_numpy(mask.astype(np.float32))[None, None], shape)[0, 0].numpy()
                path = output/f'{key}-{grid}-target.npy'
                np.save(path, target, allow_pickle=False)
                entry[f'{grid}_target_path'], entry[f'{grid}_target_sha256'] = str(path), sha256(path)
        with Image.open(row['path']) as rgb:
            entry['morphology_score'] = morphology_score(rgb, p['config']['methods']['morphology'])
        index[identity] = entry
        if i % 100 == 0 or i+1 == len(p['pool_ids']):
            print('CACHE', i+1, '/', len(p['pool_ids']), flush=True)
    write_json(output/'index.json', {'protocol_identity': p['identity'], 'rows': index,
                                    'all_1914_native_coarse_features_exact': True,
                                    'reserved_normal_calibration_decoded': False,
                                    'official_test_decoded': False})
    write_json(root/'cache-complete.json', {'protocol_identity': p['identity'],
               'index_sha256': sha256(output/'index.json'), 'native_parity_count': len(index)})


class FeatureStore:
    def __init__(self, root, p):
        marker = read_json(root/'cache-complete.json')
        if marker['protocol_identity'] != p['identity'] or sha256(root/'features/index.json') != marker['index_sha256']:
            raise ValueError('Feature index changed or cache incomplete')
        self.index = read_json(root/'features/index.json')
        if self.index['protocol_identity'] != p['identity']:
            raise ValueError('Feature manifest mismatch')
        self.checked = set()

    def array(self, identity, kind):
        entry = self.index['rows'][identity]
        path = entry[kind+'_path']
        if path not in self.checked:
            if sha256(path) != entry[kind+'_sha256']:
                raise ValueError('Feature/ROI/target cache changed')
            self.checked.add(path)
        return np.load(path, mmap_mode='r', allow_pickle=False)

    def feature(self, identity, cfg):
        return self.array(identity, cfg['grid'])[:cfg['channels']].astype(np.float32)

    def roi(self, identity, cfg):
        return np.array(self.array(identity, cfg['grid']+'_roi'), dtype=bool)

    def target(self, identity, cfg):
        return np.array(self.array(identity, cfg['grid']+'_target'), dtype=np.float32)


def partition(p, by, train_folds, *, salt):
    available = [by[i] for i in p['pool_ids'] if p['folds'][i] in train_folds]
    normals = sorted([r for r in available if not r['label']], key=lambda r: digest([p['config']['seed'], salt, r['group']]))
    # Audited KSDD2 groups are unique. Keep this guard explicit if data change.
    if len({r['group'] for r in normals}) != len(normals):
        raise ValueError('This fixed-count protocol requires one original per normal group')
    ncal = p['config']['inner_normal_calibration_count']
    calibration = normals[:ncal]
    fitting = normals[ncal:ncal+p['config']['normal_fit_cap']]+sorted([r for r in available if r['label']], key=lambda r: r['id'])
    if len(calibration) != ncal or sum(not r['label'] for r in fitting) != p['config']['normal_fit_cap']:
        raise ValueError('Insufficient normals for disjoint fitting and calibration')
    allowed = {r['id'] for r in available}
    guard(fitting, allowed_ids=allowed, forbidden_ids={r['id'] for r in calibration})
    guard(calibration, allowed_ids=allowed, forbidden_ids={r['id'] for r in fitting})
    return fitting, calibration


class Head(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(cfg['channels'], cfg['hidden'], 1), nn.ReLU(),
                                     nn.Dropout2d(cfg['dropout']), nn.Conv2d(cfg['hidden'], 1, 1))

    def forward(self, x):
        return self.network(x)[:, 0]


def pool(raw, roi, rule):
    scores = []
    for values, valid in zip(raw, roi):
        selected = values[valid]
        if not len(selected) or not torch.isfinite(selected).all():
            raise ValueError('Invalid score map')
        k = min(3, len(selected)) if rule == 'top3' else max(1, math.ceil(.001*len(selected)))
        scores.append(selected.topk(k).values.mean())
    return torch.stack(scores)


def normalization(store, fitting, cfg):
    c = cfg['channels']
    sums, squares, count = torch.zeros(c, dtype=torch.float64), torch.zeros(c, dtype=torch.float64), 0
    for row in fitting:
        if row['label']:
            continue
        x = torch.from_numpy(store.feature(row['id'], cfg))[:, store.roi(row['id'], cfg)].double()
        sums += x.sum(1); squares += x.square().sum(1); count += x.shape[1]
    mean = (sums/count).float().reshape(1, -1, 1, 1)
    std = (squares/count-(sums/count).square()).clamp_min(.01).sqrt().float().reshape(1, -1, 1, 1)
    return mean, std


def train_model(directory, store, fitting, cfg, seed):
    family = cfg['family']
    start = time.perf_counter()
    if family == 'morphology':
        state = {'family': family}
        history = []
    else:
        mean, std = normalization(store, fitting, cfg)
        state = {'mean': mean, 'std': std, 'family': family}
        torch.manual_seed(seed)
        if family == 'supervised':
            head = Head(cfg)
            optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=cfg['weight_decay'])
            generator = torch.Generator().manual_seed(seed)
            history = []
            for epoch in range(cfg['epochs']):
                head.train()
                total = 0.
                for part in torch.randperm(len(fitting), generator=generator).split(4):
                    rows = [fitting[int(i)] for i in part]
                    features = torch.from_numpy(np.stack([store.feature(r['id'], cfg) for r in rows]))
                    roi = torch.from_numpy(np.stack([store.roi(r['id'], cfg) for r in rows]))
                    target = torch.from_numpy(np.stack([store.target(r['id'], cfg) if r['label'] else np.zeros(roi.shape[-2:], np.float32) for r in rows]))
                    labels = torch.tensor([float(r['label']) for r in rows])
                    raw = head((features-mean)/std)
                    bce = F.binary_cross_entropy_with_logits(raw, target, pos_weight=torch.tensor(8.), reduction='none')
                    probability = raw.sigmoid()
                    pt = probability*target+(1-probability)*(1-target)
                    pixel = ((bce*(1-pt).square()*roi).sum((1, 2))/roi.sum((1, 2))).mean()
                    image = F.binary_cross_entropy_with_logits(pool(raw, roi, cfg['pool']), labels)
                    dice = 1-(2*(probability*target*roi).sum((1, 2))+1)/((probability*roi).sum((1, 2))+(target*roi).sum((1, 2))+1)
                    dice = dice[labels.bool()].mean() if labels.any() else raw.sum()*0.
                    loss = pixel+image+cfg['dice_weight']*dice
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite loss')
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    nn.utils.clip_grad_norm_(head.parameters(), 5.); optimizer.step()
                    total += float(loss.detach())*len(rows)
                history.append({'epoch': epoch+1, 'loss': total/len(fitting)})
                if epoch % 5 == 0 or epoch+1 == cfg['epochs']:
                    print('TRAIN', directory.name, epoch+1, history[-1]['loss'], flush=True)
            state['head'] = head.state_dict()
        else:
            patches, rng = [], np.random.default_rng(seed)
            for row in fitting:
                if row['label']:
                    continue
                x = (torch.from_numpy(store.feature(row['id'], cfg))[None]-mean)/std
                values = x[0, :, store.roi(row['id'], cfg)].T
                selected = rng.choice(len(values), min(cfg['normal_patches_per_image'], len(values)), replace=False)
                patches.append(values[selected])
            values = torch.cat(patches)
            history = []
            if family == 'patchcore':
                projection = torch.from_numpy(rng.standard_normal((cfg['channels'], cfg['projection_dimensions']), dtype=np.float32))/math.sqrt(cfg['projection_dimensions'])
                selected = farthest_points(values@projection, min(cfg['bank_vectors'], len(values)))
                state['bank'] = values[selected].clone()
            elif family == 'pca':
                center = values.double().mean(0)
                x = values.double()-center
                eigenvalues, eigenvectors = torch.linalg.eigh(x.T@x/(len(x)-1))
                state['center'] = center.float()
                state['basis'] = eigenvectors[:, -cfg['rank']:].float()
                state['eigenvalues'] = eigenvalues.float()
            else:
                raise ValueError('Unknown model family')
    torch.save(state, directory/'model.pt')
    write_json(directory/'training.json', {'fitting_ids': [r['id'] for r in fitting],
               'normalization_ids': [r['id'] for r in fitting if not r['label']] if family != 'morphology' else [],
               'learning_labels_used': family == 'supervised', 'normal_only': family in ('patchcore', 'pca'),
               'actual_learning_ids': [r['id'] for r in fitting if family == 'supervised' or (not r['label'] and family != 'morphology')],
               'history': history, 'fit_seconds': time.perf_counter()-start, 'seed': seed, 'configuration': cfg,
               'no_evaluation_stopping_or_mining': True, 'model_sha256': sha256(directory/'model.pt')})
    return state


def scorer(state, cfg):
    if cfg['family'] == 'supervised':
        head = Head(cfg); head.load_state_dict(state['head']); head.eval()
        return head
    return None


@torch.inference_mode()
def score(store, identity, cfg, state, head):
    if cfg['family'] == 'morphology':
        return store.index['rows'][identity]['morphology_score']
    roi = torch.from_numpy(store.roi(identity, cfg))
    x = (torch.from_numpy(store.feature(identity, cfg))[None]-state['mean'])/state['std']
    if head is not None:
        return float(pool(head(x), roi[None], cfg['pool'])[0])
    values = x[0, :, roi].T.contiguous()
    if cfg['family'] == 'pca':
        values = values-state['center']
        residual = (values.square().sum(1)-(values@state['basis']).square().sum(1)).clamp_min(0)
        return float(residual.sqrt().max())
    nearest = []
    for chunk in values.split(cfg['query_chunk']):
        nearest.append(torch.cdist(chunk, state['bank']).min(1).values)
    return float(torch.cat(nearest).max())


def result(directory, name, rows, values, calibration, threshold, p):
    if set(values) != {r['id'] for r in rows} or any(not np.isfinite(s) for s in values.values()):
        raise ValueError('Incomplete/nonfinite scores')
    predictions = [{'id': r['id'], 'label': r['label'], 'group': r['group'], 'score': values[r['id']],
                    'status': 'scored', 'flagged': values[r['id']] > threshold['threshold']} for r in rows]
    metrics = evaluate(predictions, {r['id']: r['label'] for r in rows})
    small = [r for r in predictions if r['label'] and p['positive_source_area_fraction'][r['id']] <= .005]
    report = {'protocol_identity': p['identity'], 'predictions': predictions, 'metrics': metrics,
              'threshold_record': threshold, 'calibration_scores': calibration,
              'small_defects': {'detected': sum(r['flagged'] for r in small), 'total': len(small)},
              'claim': p['config']['claim']}
    write_json(directory/name, report)
    print('RESULT', name, metrics['counts'], report['small_defects'], flush=True)
    return report


def fit_and_score(root, stage, fold, method):
    p, _, by = context(root)
    cfg = p['config']['methods'][method]
    if stage == 'inner':
        train_folds = [fold]
        validation_folds = [f for f in range(3) if f != fold]
        directory = root/f'inner-{method}-{fold}'
        salt = f'inner-{fold}'
    else:
        selected = load_selection(root, p)
        method = selected['outer_choices'][str(fold)]['method']
        cfg = p['config']['methods'][method]
        train_folds = [f for f in range(3) if f != fold]
        validation_folds = [fold]
        directory = root/f'outer-{fold}'
        salt = f'outer-{fold}'
    directory.mkdir(exist_ok=False)
    fitting, calibration_rows = partition(p, by, train_folds, salt=salt)
    forbidden = {i for i in p['pool_ids'] if p['folds'][i] not in train_folds}
    guard(fitting+calibration_rows, forbidden_ids=forbidden | set(p['reserved_normal_calibration_ids']))
    write_json(directory/'plan.json', {'stage': stage, 'fold': fold, 'method': method,
               'training_folds': train_folds, 'validation_folds': validation_folds,
               'fitting_ids': [r['id'] for r in fitting], 'calibration_ids': [r['id'] for r in calibration_rows],
               'excluded_fold_ids': sorted(forbidden), 'official_test_access': False,
               'selection_identity': load_selection(root, p)['identity'] if stage == 'outer' else None})
    store = FeatureStore(root, p)
    state = train_model(directory, store, fitting, cfg, p['config']['seed']+fold+(100 if stage == 'outer' else 0))
    state = torch.load(directory/'model.pt', weights_only=True)
    head = scorer(state, cfg)
    calibration = [{'id': r['id'], 'score': score(store, r['id'], cfg, state, head)} for r in calibration_rows]
    threshold = calibrate([r['score'] for r in calibration], alpha=p['config']['calibration_alpha'])
    if cfg['family'] == 'supervised':
        fitting_values = {r['id']: score(store, r['id'], cfg, state, head) for r in fitting}
        result(directory, 'fitting-diagnostic.json', fitting, fitting_values, calibration, threshold, p)
    for validation_fold in validation_folds:
        rows = [by[i] for i in p['pool_ids'] if p['folds'][i] == validation_fold]
        guard(rows, forbidden_ids={r['id'] for r in fitting+calibration_rows})
        values = {r['id']: score(store, r['id'], cfg, state, head) for r in rows}
        result(directory, f'eval-{validation_fold}.json', rows, values, calibration, threshold, p)
        if stage == 'outer':
            restored_state = torch.load(directory/'model.pt', weights_only=True)
            restored_head = scorer(restored_state, cfg)
            replay = {r['id']: score(store, r['id'], cfg, restored_state, restored_head) for r in reversed(rows)}
            difference = max(abs(values[i]-replay[i]) for i in values)
            write_json(directory/'replay.json', {'max_absolute_score_difference': difference,
                       'all_scores_exact': difference == 0., 'image_count': len(rows),
                       'scope': 'reloaded model, reverse cached-feature order; final native pipeline checked separately'})
            if difference != 0.:
                raise ValueError('Outer cached-feature head replay differs')
    write_json(directory/'complete.json', {'method': method, 'stage': stage, 'fold': fold,
               'model_sha256': sha256(directory/'model.pt'), 'protocol_identity': p['identity']})


def select_method(candidates):
    """Only the explicitly supplied inner metrics may influence this decision."""
    def key(item):
        name, records = item
        worst_fpr = max(r['metrics']['accepted_normal_fpr']['rate'] for r in records)
        recall = [r['metrics']['recall']['rate'] for r in records]
        small_total = sum(r['small_defects']['total'] for r in records)
        small = sum(r['small_defects']['detected'] for r in records)/small_total if small_total else 0.
        mean_fpr = sum(r['metrics']['accepted_normal_fpr']['rate'] for r in records)/len(records)
        return (worst_fpr >= .1, worst_fpr if worst_fpr >= .1 else 0.,
                -min(recall), -sum(recall)/len(recall), -small, mean_fpr, name)
    if not candidates:
        raise ValueError('No complete candidate can be selected')
    ranked = sorted(candidates.items(), key=key)
    return ranked[0][0], [{'method': name, 'selection_key': list(key((name, records)))} for name, records in ranked]


def load_selection(root, p):
    value = read_json(root/'selection.json')
    identity = value.pop('identity')
    if digest(value) != identity or value['protocol_identity'] != p['identity']:
        raise ValueError('Frozen selection changed')
    value['identity'] = identity
    for choice in value['outer_choices'].values():
        for files in choice['input_files'].values():
            for file in files:
                if sha256(file['path']) != file['sha256']:
                    raise ValueError('Inner selection input changed')
    return value


def select(root):
    p, _, _ = context(root)
    outer_choices, final_inputs, failures = {}, {}, []
    complete_methods = []
    for method in p['config']['methods']:
        if all((root/f'inner-{method}-{f}/complete.json').exists() for f in range(3)):
            complete_methods.append(method)
        else:
            failures.append(method)
    for outer in range(3):
        allowed = [f for f in range(3) if f != outer]
        candidates, files = {}, {}
        for method in complete_methods:
            records, paths = [], []
            for train in allowed:
                validation = next(f for f in allowed if f != train)
                path = root/f'inner-{method}-{train}/eval-{validation}.json'
                record = read_json(path)
                touched = {r['id'] for r in record['predictions']+record['calibration_scores']}
                touched |= set(read_json(path.parent/'training.json')['actual_learning_ids'])
                if any(p['folds'][i] == outer for i in touched):
                    raise ValueError('Outer fold leaked into method selection')
                records.append(record); paths.append({'path': str(path), 'sha256': sha256(path)})
            candidates[method], files[method] = records, paths
            final_inputs.setdefault(method, []).extend(records)
        method, ranking = select_method(candidates)
        outer_choices[str(outer)] = {'method': method, 'ranking': ranking, 'input_files': files,
                                     'excluded_outer_fold': outer}
    final_method, final_ranking = select_method(final_inputs)
    record = {'protocol_identity': p['identity'], 'outer_choices': outer_choices,
              'final_method': final_method, 'final_ranking': final_ranking,
              'incomplete_methods_not_silently_zeroed': failures,
              'created_before_any_outer_prediction': not any(root.glob('outer-*/eval-*.json')),
              'final_selection_uses_inner_comparisons_only': True}
    if not record['created_before_any_outer_prediction']:
        raise ValueError('Selection must precede all outer predictions')
    record['identity'] = digest(record)
    write_json(root/'selection.json', record)
    print('SELECTED', {f: r['method'] for f, r in outer_choices.items()}, 'FINAL', final_method, flush=True)


def confidence(metrics):
    n, tp = metrics['recall']['denominator'], metrics['counts']['TP']
    normals, fp = metrics['false_flags_all_normals']['denominator'], metrics['counts']['FP']
    lower = float(binomtest(tp, n, alternative='greater').proportion_ci(.975, method='exact').low)
    upper = float(binomtest(fp, normals, alternative='less').proportion_ci(.975, method='exact').high)
    return {'recall_exact_95': list(binomtest(tp, n).proportion_ci()),
            'fpr_exact_95': list(binomtest(fp, normals).proportion_ci()),
            'simultaneous_95_recall_lower': lower, 'simultaneous_95_fpr_upper': upper,
            'point_goal_met': tp/n >= .95 and fp/normals < .1 and metrics['coverage']['rate'] == 1,
            'confidence_goal_met': lower >= .95 and upper < .1,
            'scope': 'internal nested CV; prior data exposure and unknown product identity prevent independent production proof'}


class NativeStore:
    """A single original image; the scorer receives pixels/features, never truth."""
    def __init__(self, identity, coarse, fine, coarse_roi, fine_roi, morphology):
        self.coarse, self.fine = coarse, fine
        self.coarse_roi, self.fine_roi = coarse_roi, fine_roi
        self.index = {'rows': {identity: {'morphology_score': morphology}}}

    def feature(self, identity, cfg):
        return (self.coarse if cfg['grid'] == 'coarse' else self.fine)[:cfg['channels']]

    def roi(self, identity, cfg):
        return self.coarse_roi if cfg['grid'] == 'coarse' else self.fine_roi


@torch.inference_mode()
def native_score(adapter, row, input_config, cfg, state, head, morphology_cfg):
    prepared = image_array(row, input_config)
    tensors = adapter.model.feature_extractor(adapter._tensor(prepared.rgb_float32))
    f1, f2, f3 = (tensors[f'layer{n}'] for n in (1, 2, 3))
    coarse = F.avg_pool2d(torch.cat([f2, F.interpolate(f3, size=f2.shape[-2:], mode='nearest')], 1),
                          3, stride=1, padding=1)[0].half().float().numpy()
    fine = torch.cat([f1, F.interpolate(f2, size=f1.shape[-2:], mode='nearest'),
                      F.interpolate(f3, size=f1.shape[-2:], mode='nearest')], 1)[0].half().float().numpy()
    roi = torch.from_numpy(prepared.valid_roi.astype(np.float32))[None, None]
    coarse_roi = F.interpolate(roi, size=coarse.shape[-2:], mode='nearest')[0, 0].bool().numpy()
    fine_roi = F.interpolate(roi, size=fine.shape[-2:], mode='nearest')[0, 0].bool().numpy()
    with Image.open(row['path']) as rgb:
        morphology = morphology_score(rgb, morphology_cfg) if cfg['family'] == 'morphology' else 0.
    native = NativeStore(row['id'], coarse, fine, coarse_roi, fine_roi, morphology)
    return score(native, row['id'], cfg, state, head)


@torch.inference_mode(False)
def final(root):
    """Fit the inner-selected recipe and calibrate it before outer unblinding."""
    from component_inspector.vision.spike_worker import pretrained_adapter
    p, m, by = context(root)
    selection = load_selection(root, p)
    if any(root.glob('outer-*/eval-*.json')):
        raise ValueError('Final candidate must be frozen before any outer result')
    method = selection['final_method']
    cfg = p['config']['methods'][method]
    directory = root/'final'
    directory.mkdir(exist_ok=False)
    normals = sorted([by[i] for i in p['pool_ids'] if not by[i]['label']],
                     key=lambda r: digest([p['config']['seed'], 'final', r['group']]))
    fitting = normals[:p['config']['normal_fit_cap']]+[by[i] for i in p['pool_ids'] if by[i]['label']]
    guard(fitting, forbidden_ids=set(p['reserved_normal_calibration_ids']))
    write_json(directory/'plan.json', {'method': method, 'selection_identity': selection['identity'],
               'fitting_ids': [r['id'] for r in fitting],
               'reserved_normal_calibration_ids': p['reserved_normal_calibration_ids'],
               'outer_results_accessed': False, 'official_test_accessed': False,
               'claim': 'candidate for future independent confirmation; no performance claim from final training or calibration'})
    store = FeatureStore(root, p)
    state = train_model(directory, store, fitting, cfg, p['config']['seed']+200)
    state = torch.load(directory/'model.pt', weights_only=True)
    head = scorer(state, cfg)
    adapter = pretrained_adapter('ksdd2_surface', m['config'], Path(p['weights']))
    calibration, times = [], []
    with torch.inference_mode():
        for i, identity in enumerate(p['reserved_normal_calibration_ids']):
            row = by[identity]
            guard([row], allowed_ids=set(p['reserved_normal_calibration_ids']),
                  forbidden_ids={r['id'] for r in fitting})
            if row['label'] != 0:
                raise ValueError('Positive in reserved normal calibration')
            tick = time.perf_counter()
            value = native_score(adapter, row, m['config'], cfg, state, head,
                                 p['config']['methods']['morphology'])
            times.append(time.perf_counter()-tick)
            calibration.append({'id': identity, 'score': value})
            if i % 100 == 0 or i+1 == len(p['reserved_normal_calibration_ids']):
                print('FINAL NORMAL CALIBRATION', i+1, '/', len(p['reserved_normal_calibration_ids']), flush=True)
        threshold = calibrate([r['score'] for r in calibration], alpha=p['config']['calibration_alpha'])
        samples = ([by[i] for i in p['pool_ids'] if by[i]['label']][:8]
                   +[by[i] for i in p['pool_ids'] if not by[i]['label']][:8])
        differences, decisions = [], []
        for row in samples:
            cached_value = score(store, row['id'], cfg, state, head)
            native_value = native_score(adapter, row, m['config'], cfg, state, head,
                                        p['config']['methods']['morphology'])
            differences.append(abs(native_value-cached_value))
            decisions.append((native_value > threshold['threshold']) == (cached_value > threshold['threshold']))
    if max(differences) > 1e-5 or not all(decisions):
        raise ValueError('Saved-cache and native final pipeline parity failed')
    record = {'method': method, 'protocol_identity': p['identity'], 'selection_identity': selection['identity'],
              'model_sha256': sha256(directory/'model.pt'), 'threshold_record': threshold,
              'calibration_scores': calibration, 'native_parity_sample_ids': [r['id'] for r in samples],
              'max_native_cache_score_difference': max(differences), 'all_native_cache_decisions_equal': all(decisions),
              'native_calibration_pipeline_p95_seconds': float(np.percentile(times, 95)),
              'configuration': cfg, 'no_outer_or_official_test_selection': True,
              'performance_on_new_factory_images': 'unmeasured; no separate batch available'}
    record['identity'] = digest(record)
    write_json(directory/'frozen.json', record)
    print('FINAL FROZEN', method, record['model_sha256'], threshold, flush=True)


@torch.inference_mode()
def summarize(root):
    p, _, by = context(root)
    selection = load_selection(root, p)
    frozen = read_json(root/'final/frozen.json')
    if (frozen['selection_identity'] != selection['identity'] or
            sha256(root/'final/model.pt') != frozen['model_sha256']):
        raise ValueError('Final candidate changed after inner selection')
    predictions, folds = [], {}
    for fold in range(3):
        directory = root/f'outer-{fold}'
        record = read_json(directory/f'eval-{fold}.json')
        plan, training = read_json(directory/'plan.json'), read_json(directory/'training.json')
        excluded = set(plan['excluded_fold_ids'])
        if (excluded & set(training['actual_learning_ids']) or excluded & set(plan['calibration_ids'])
                or plan['selection_identity'] != selection['identity']):
            raise ValueError('Outer provenance/selection violation')
        folds[str(fold)] = {'method': plan['method'], 'metrics': record['metrics'],
                            'confidence': confidence(record['metrics']), 'small_defects': record['small_defects']}
        predictions.extend([{**r, 'outer_fold': fold, 'method': plan['method']} for r in record['predictions']])
    if len(predictions) != len(p['pool_ids']) or {r['id'] for r in predictions} != set(p['pool_ids']):
        raise ValueError('Nested predictions not exactly once per original')
    metrics = evaluate(predictions, {i: by[i]['label'] for i in p['pool_ids']})
    # Different independently calibrated models supply outer scores. Their raw
    # logit/distance scales cannot be pooled into one meaningful ranking curve.
    metrics['ranking_scored_only'] = {'population': 'not pooled: different outer-model score scales', 'auroc': None, 'average_precision': None}
    small = [r for r in predictions if r['label'] and p['positive_source_area_fraction'][r['id']] <= .005]
    record = {'protocol_identity': p['identity'], 'selection_identity': selection['identity'],
              'outer_folds': folds, 'nested_metrics': metrics, 'nested_confidence': confidence(metrics),
              'small_defects': {'detected': sum(r['flagged'] for r in small), 'total': len(small)},
              'nested_predictions': predictions, 'final_inner_selected_method': selection['final_method'],
              'final_model_identity': frozen['identity'], 'final_model_sha256': frozen['model_sha256'],
              'no_official_test_evaluation': True, 'no_new_independent_confirmation': True,
              'selected_pipeline_not_single_method_estimate': 'outer method selection may differ by fold; this estimates the predeclared selection procedure',
              'all_history_preserved': True}
    write_json(root/'nested-results.json', record)
    print('NESTED', metrics['counts'], record['nested_confidence'], record['small_defects'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['freeze', 'cache', 'inner', 'select', 'final', 'outer', 'summarize'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path, default=Path('components/outputs/recall-round1'))
    parser.add_argument('--weights', type=Path, default=Path('.cache/recall-downloads/model.safetensors'))
    parser.add_argument('--fold', type=int, choices=[0, 1, 2])
    parser.add_argument('--method', default='coarse_control')
    args = parser.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    root = args.root.resolve()
    if args.phase == 'freeze': freeze(root, args.parent.resolve(), args.weights.resolve())
    elif args.phase == 'cache': cache(root)
    elif args.phase == 'select': select(root)
    elif args.phase == 'final': final(root)
    elif args.phase == 'summarize': summarize(root)
    else: fit_and_score(root, args.phase, args.fold, args.method)


if __name__ == '__main__':
    main()
