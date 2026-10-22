"""Training/scoring alignment experiment after a reproducible seed collapse."""
import argparse
import gc
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from component_inspector.data.acquire import sha256
from component_inspector.research.ksdd2_round1 import DefectHead, guard, image_array, load_manifest, rows_for, write_json
from component_inspector.research.ksdd2_round2 import cache_index, cached, context, focal_loss, read_json, roi_array, training_rows
from component_inspector.research.ksdd2_round2_followup import score_head


def fixed_pool(raw, roi):
    values = []
    for row, valid in zip(raw, roi):
        selected = row[valid]
        if not len(selected):
            raise ValueError('Empty scoring ROI')
        values.append(selected.topk(max(1, math.ceil(.001*len(selected)))).values.mean())
    return torch.stack(values)


def aligned_objective(head, features, roi, target, labels):
    # Optimize the exact positive-oriented score used at inference. The old
    # learned classifier could reward decreasing the largest defect logits.
    raw = head.segment(features)[:, 0]
    pixel = (focal_loss(raw, target)*roi).sum()/roi.sum()
    return pixel+F.binary_cross_entropy_with_logits(fixed_pool(raw, roi), labels)


def fit(head, rows, base, index, mean, std, *, seed, epochs, learning_rate, label):
    guard(rows)
    assert len({r['id'] for r in rows}) == len(rows)
    assert all(r['v2_role'] in {'normal_fit', 'positive_train'} for r in rows)
    shape = cache_index(base)['shape'][-2:]
    roi = torch.from_numpy(np.stack([roi_array(base, index[r['id']]) for r in rows]))
    target = torch.from_numpy(np.stack([
        np.load(base/f"{index[r['id']]['key']}-target.npy", allow_pickle=False)
        if r['label'] else np.zeros(shape, np.float32) for r in rows]))
    labels = torch.tensor([float(r['label']) for r in rows])
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(head.segment.parameters(), lr=learning_rate, weight_decay=.0001)
    history = []
    start = time.perf_counter()
    for epoch in range(epochs):
        head.train()
        order = torch.randperm(len(rows), generator=generator)
        total = 0.
        for offset in range(0, len(rows), 4):
            ids = order[offset:offset+4]
            features = torch.from_numpy(np.stack([cached(base, index[rows[int(i)]['id']]) for i in ids]))
            optimizer.zero_grad(set_to_none=True)
            loss = aligned_objective(head, (features-mean)/std, roi[ids], target[ids], labels[ids])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.segment.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach())*len(ids)
        history.append({'epoch': epoch+1, 'training_loss': total/len(rows)})
        if epoch % 5 == 0 or epoch+1 == epochs:
            print(label, epoch+1, history[-1]['training_loss'], flush=True)
    return {'fitting_ids': [r['id'] for r in rows], 'epochs': epochs,
            'learning_rate': learning_rate, 'fit_seconds': time.perf_counter()-start, 'history': history}


def train(root, seed):
    parent, manifest, cfg = context(root)
    settings = cfg['aligned_training']
    method = 'aligned' if seed == 4217 else f'aligned_seed{seed}'
    output = root/method
    output.mkdir(exist_ok=False)
    base = parent/'features'
    metadata = cache_index(base)
    index = metadata['rows']
    rows = training_rows(manifest)
    write_json(output/'training-plan.json', {'seed': seed, 'configuration': settings,
               'base_fitting_ids': [r['id'] for r in rows], 'official_test_access': False,
               'evaluation_for_stopping_or_tuning': False})
    channels, h, w = metadata['shape']
    sums, squares = torch.zeros(channels), torch.zeros(channels)
    for row in rows:
        f = torch.from_numpy(cached(base, index[row['id']]))
        sums += f.sum((1, 2)); squares += f.square().sum((1, 2))
    count = len(rows)*h*w
    mean = (sums/count).reshape(1, -1, 1, 1)
    std = (squares/count-(sums/count).square()).clamp_min(.01).sqrt().reshape(1, -1, 1, 1)
    torch.manual_seed(seed)
    head = DefectHead(channels, 32)
    head.classify.requires_grad_(False)
    initial = fit(head, rows, base, index, mean, std, seed=seed, epochs=settings['base_epochs'],
                  learning_rate=.001, label=method+' base')
    torch.save({'head': head.state_dict(), 'mean': mean, 'std': std}, output/'base-head.pt')
    head.eval()
    original = set(manifest['original_fit_ids'])
    candidates = [r for r in rows_for(manifest, 'normal_fit') if r['id'] not in original]
    ranked = []
    with torch.inference_mode():
        for row in candidates:
            entry = index[row['id']]
            feature = torch.from_numpy(cached(base, entry))[None]
            roi = torch.from_numpy(roi_array(base, entry))[None]
            score = fixed_pool(head.segment((feature-mean)/std)[:, 0], roi)
            ranked.append((float(score[0]), row))
    hard = [row for _, row in sorted(ranked, key=lambda x: (-x[0], x[1]['id']))[:settings['hard_normal_count']]]
    for row in hard:
        assert row['v2_role'] == 'normal_fit' and row['id'] not in original
        if image_array(row, manifest['config'], mask=True).any():
            raise ValueError('Hard normal has a positive provider mask')
    refinement = fit(head, rows+hard, base, index, mean, std, seed=seed, epochs=settings['refinement_epochs'],
                     learning_rate=.0002, label=method+' refinement')
    torch.save({'head': head.state_dict(), 'mean': mean, 'std': std}, output/'head.pt')
    write_json(output/'model.json', {'channels': channels, 'hidden': 32, 'seed': seed,
               'cache': str(base), 'fine': False, 'focal': True, 'lighting': False,
               'image_objective': 'fixed positive-oriented top 0.1 percent logit BCE',
               'learned_classifier_used_for_training_or_inference': False})
    write_json(output/'training.json', {'base': initial, 'refinement': refinement,
               'fitting_ids': refinement['fitting_ids'], 'hard_normal_ids': [r['id'] for r in hard],
               'objective': 'weighted focal pixel loss plus BCE of exact inference pooling',
               'no_evaluation_based_stopping': True})
    del head
    gc.collect()
    score_head(root, method, method, strategy='R05/R06 correction: monotone, aligned training and inference')
    write_json(root/f'{method}-complete.json', {'seed': seed, 'source_sha256': sha256(Path(__file__)),
               'head_sha256': sha256(output/'head.pt')})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=['prepare', 'train'], required=True)
    parser.add_argument('--parent', type=Path, default=Path('components/outputs/recall-round1'))
    parser.add_argument('--seed', type=int, choices=[4217, 4218, 4219])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    root = args.root.resolve()
    if args.phase == 'prepare':
        root.mkdir(parents=True, exist_ok=False)
        manifest = load_manifest(args.parent.resolve())
        config = read_json(Path('components/configs/recall_round2.json'))
        config['scope'] = ['R05/R06 training-scoring alignment after seed collapse']
        config['selection'] = 'predeclared canonical seed 4217; report both fixed repeat seeds without replacing canonical result'
        config['aligned_training'] = {'base_epochs': 60, 'refinement_epochs': 12, 'hard_normal_count': 128,
                                     'image_pool': 'top_0.1_percent', 'fixed_seeds': [4217, 4218, 4219],
                                     'seed_policy': '4217 canonical; both repeats always reported; no best-seed selection'}
        write_json(root/'protocol.json', {'parent': str(args.parent.resolve()), 'manifest_identity': manifest['identity'],
                   'config': config, 'source_sha256': sha256(Path(__file__)),
                   'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   'reason': 'reproducible seed4219 collapse; learned negative maximum coefficient conflicts with positive-oriented inference pooling',
                   'label_exposure': 'same already observed development split; not independent confirmation'})
        write_json(root/'selected-for-stability.json', {'selected_result': 'aligned',
                   'recipe': {'model': 'aligned', 'rule': 'top_0.1_percent', 'fusion': None},
                   'selection': 'predeclared canonical seed before new training; repeats cannot replace it'})
    else:
        if args.seed is None:
            raise ValueError('Training requires a fixed seed')
        train(root, args.seed)


if __name__ == '__main__':
    main()
