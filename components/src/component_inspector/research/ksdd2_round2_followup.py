"""Finite improvements and repeated-initialization checks after fixed round 2.

No evaluation image becomes training data. Adaptive choices are explicitly
selection on the already observed development set, never new confirmation.
"""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from component_inspector.data.acquire import sha256
from component_inspector.research.ksdd2_round1 import DefectHead, image_array, rows_for, write_json
from component_inspector.research.ksdd2_round2 import (
    cache_index, cached, cdf_transform, context, digest, focal_loss, model_load,
    read_json, result, roi_array, score_rules, training_rows,
)


def source_model(root, name):
    parent, _, _ = context(root)
    path = parent/'supervised' if name == 'round1/supervised' else root/name
    head, state, meta = model_load(path)
    base = Path(meta.get('cache', parent/'features'))
    return head, state, meta, path, base


@torch.inference_mode()
def score_head(root, model, method, rule='top_0.1_percent', *, strategy='R06 applied to a learned candidate'):
    _, m, _ = context(root)
    head, state, _, path, base = source_model(root, model)
    index = cache_index(base)['rows']
    cal, score = {}, {}
    for role in ['normal_calibration', 'normal_evaluation', 'positive_evaluation']:
        for row in rows_for(m, role):
            entry = index[row['id']]
            f = torch.from_numpy(cached(base, entry))[None]
            roi = torch.from_numpy(roi_array(base, entry))
            raw, image = head((f-state['mean'])/state['std'], roi[None])
            value = float(image[0]) if rule == 'image' else score_rules(raw[0], roi)[rule]
            (cal if role == 'normal_calibration' else score)[row['id']] = value
    return result(root, method, m, score, cal, details={'strategy': strategy, 'source_model': model,
                  'rule': rule, 'model_sha256': sha256(path/'head.pt'), 'no_evaluation_refitting': True})


def candidate_recipes(root):
    mappings = {}
    for p in root.glob('*/result.json'):
        value = read_json(p)
        if 'metrics' not in value or p.parent.name.startswith('stability') or '_seed' in p.parent.name:
            continue
        name = p.parent.name
        details = value.get('details', {})
        if 'head_recipe' in details:
            recipe = {**details['head_recipe'], 'fusion': details['fixed_rule']}
        elif 'source_model' in details:
            recipe = {'model': details['source_model'], 'rule': details['rule'], 'fusion': None}
        elif name.startswith('aggregation_'):
            recipe = {'model': 'round1/supervised', 'rule': details['rule'], 'fusion': None}
        elif (root/name/'model.json').exists():
            recipe = {'model': name, 'rule': details.get('rule', 'image'), 'fusion': None}
        elif name.startswith('ensemble_'):
            model = details['parents'][0]
            recipe = {'model': 'round1/supervised' if model == 'supervised' else model,
                      'rule': 'image', 'fusion': details['fixed_rule']}
        else:
            continue
        mappings[name] = {'result': value, 'recipe': recipe}
    return mappings


def best(root, *, single_head=False):
    choices = [(name, value) for name, value in candidate_recipes(root).items()
               if value['result']['metrics']['accepted_normal_fpr']['rate'] < .10
               and (not single_head or value['recipe']['fusion'] is None)]
    if not choices:
        raise ValueError('No eligible learned development candidate')
    return max(choices, key=lambda v: (v[1]['result']['metrics']['recall']['rate'],
                                      -v[1]['result']['metrics']['accepted_normal_fpr']['rate'],
                                      v[1]['recipe']['fusion'] is None))


def score_heads(root):
    write_json(root/'score-heads-plan.json', {'rule': 'top_0.1_percent',
               'rationale': 'fixed best rule from the already observed parent scoring comparison; apply identically to each new head',
               'selection_claim': 'observed development comparison'})
    for name in ['fine', 'focal', 'lighting', 'combined']:
        if (root/name/'head.pt').exists():
            score_head(root, name, name+'_top001')


def combined(root):
    from component_inspector.research.ksdd2_round2 import train
    parent, _, _ = context(root)
    baseline = read_json(parent/'supervised/result.json')['metrics']['recall']['rate']
    condition = all((root/name/'result.json').exists() and read_json(root/name/'result.json')['metrics']['recall']['rate'] >= baseline
                    and read_json(root/name/'result.json')['metrics']['accepted_normal_fpr']['rate'] < .10
                    for name in ['focal', 'lighting'])
    write_json(root/'combined-decision.json', {'run': condition, 'rule': 'both isolated changes must preserve parent recall with FPR under 10%',
                                             'parent_recall': baseline})
    if condition:
        train(root, 'combined', strategy_override='bounded combination of R05 and R08')
        score_head(root, 'combined', 'combined_top001')


def refine(root, parent_recipe, name='hard_refined', *, seed=4217):
    parent, m, cfg = context(root)
    head, state, metadata, model_path, base = source_model(root, parent_recipe['model'])
    if metadata.get('fine', False):
        write_json(root/f'{name}-decision.json', {'status': 'deferred_with_reason',
                   'reason': 'Best head uses fine features; extra normal fine-feature extraction is outside this bounded refinement. R05 focal experiment remains completed.'})
        return None
    index = cache_index(base)['rows']
    original = set(m['original_fit_ids'])
    eligible = [r for r in rows_for(m, 'normal_fit') if r['id'] not in original]
    rankings = []
    head.eval()
    with torch.inference_mode():
        for row in eligible:
            entry = index[row['id']]
            f = torch.from_numpy(cached(base, entry))[None]
            roi = torch.from_numpy(roi_array(base, entry))
            pixel, image = head((f-state['mean'])/state['std'], roi[None])
            score = float(image[0]) if parent_recipe['rule'] == 'image' else score_rules(pixel[0], roi)[parent_recipe['rule']]
            rankings.append((score, row))
    selected = [row for _, row in sorted(rankings, key=lambda x: x[0], reverse=True)[:cfg['adaptive_limit']['hard_negative_add_count']]]
    assert all(r['v2_role'] == 'normal_fit' and r['id'] not in original for r in selected)
    for row in selected:
        if image_array(row, m['config'], mask=True).any():
            raise ValueError('A mined normal has a positive provider mask')
    rows = training_rows(m)+selected
    y = torch.tensor([float(r['label']) for r in rows])
    roi = torch.from_numpy(np.stack([roi_array(base, index[r['id']]) for r in rows]))
    shape = cache_index(base)['shape'][-2:]
    target = torch.from_numpy(np.stack([np.load(base/f"{index[r['id']]['key']}-target.npy", allow_pickle=False)
                                      if r['label'] else np.zeros(shape, np.float32) for r in rows]))
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.0002, weight_decay=.0001)
    history = []
    start = time.perf_counter()
    for epoch in range(cfg['adaptive_limit']['refinement_epochs']):
        head.train()
        order = torch.randperm(len(rows), generator=generator)
        total = 0.
        for offset in range(0, len(rows), 4):
            ids = order[offset:offset+4]
            f = torch.from_numpy(np.stack([cached(base, index[rows[int(i)]['id']]) for i in ids]))
            optimizer.zero_grad(set_to_none=True)
            pixel, image = head((f-state['mean'])/state['std'], roi[ids])
            if metadata.get('focal', False):
                pixel_loss = focal_loss(pixel, target[ids])
            else:
                pixel_loss = F.binary_cross_entropy_with_logits(pixel, target[ids], pos_weight=torch.tensor(8.), reduction='none')
            loss = (pixel_loss*roi[ids]).sum()/roi[ids].sum()+F.binary_cross_entropy_with_logits(image, y[ids])
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach())*len(ids)
        history.append({'epoch': epoch+1, 'training_loss': total/len(rows)})
        print(name, 'epoch', epoch+1, 'loss', history[-1]['training_loss'], flush=True)
    output = root/name
    output.mkdir(exist_ok=False)
    torch.save({'head': head.state_dict(), 'mean': state['mean'], 'std': state['std']}, output/'head.pt')
    write_json(output/'model.json', {**metadata, 'channels': 384, 'hidden': 32, 'cache': str(base),
                                   'hard_negative_refinement': True, 'parent_recipe': parent_recipe, 'seed': seed})
    write_json(output/'training.json', {'strategy': 'R05 training-only hard negatives',
               'parent_model_sha256': sha256(model_path/'head.pt'), 'fit_seconds': time.perf_counter()-start,
               'fitting_ids': [r['id'] for r in rows], 'hard_normal_ids': [r['id'] for r in selected],
               'epochs': 12, 'learning_rate': .0002, 'augmentation_during_refinement': 'none', 'history': history})
    del optimizer, head, target
    gc.collect()
    return score_head(root, name, name, parent_recipe['rule'], strategy='R05 training-only hard-negative refinement')


def hard(root):
    name, selected = best(root, single_head=True)
    write_json(root/'hard-refinement-parent.json', {'selected_result': name, 'recipe': selected['recipe'],
               'selection': 'highest observed development recall under 10% FPR, then lower FPR'})
    refine(root, selected['recipe'])


def fuse(root, first, head_recipe, method, rule):
    parent, m, cfg = context(root)
    second = read_json(parent/'patchcore/result.json')
    ids = sorted([r['id'] for r in rows_for(m, 'normal_calibration')], key=lambda i: digest([cfg['seed'], 'ensemble_normalization', i]))
    n = round(len(ids)*.5)
    norm_ids, cal_ids = ids[:n], ids[n:]
    eval_ids = sorted(r['id'] for r in first['predictions'])
    all_ids = cal_ids+eval_ids
    z, cdf, normalization = [], [], []
    for source in [first, second]:
        cal = {r['id']: r['score'] for r in source['calibration_scores']}
        score = {r['id']: r['score'] for r in source['predictions']}
        normal = np.asarray([cal[i] for i in norm_ids])
        values = np.asarray([cal[i] if i in cal else score[i] for i in all_ids])
        median = float(np.median(normal))
        iqr = max(float(np.percentile(normal, 75)-np.percentile(normal, 25)), 1e-6)
        z.append((values-median)/iqr)
        cdf.append(cdf_transform(normal, values))
        normalization.append({'median': median, 'iqr': iqr, 'cdf_reference': normal.tolist()})
    values = np.maximum(cdf[0], cdf[1]) if rule == 'maximum_normal_cdf' else .75*z[0]+.25*z[1]
    return result(root, method, m, dict(zip(eval_ids, values[len(cal_ids):])), dict(zip(cal_ids, values[:len(cal_ids)])),
                  details={'strategy': 'R10 applied to best single-head score', 'head_recipe': head_recipe,
                           'fixed_rule': rule, 'normalization_ids': norm_ids, 'threshold_ids': cal_ids,
                           'normalizers': normalization, 'parents': [head_recipe['model'], 'round1/patchcore']})


def fusion_best(root):
    _, _, cfg = context(root)
    name, selected = best(root, single_head=True)
    write_json(root/'fusion-best-parent.json', {'selected_result': name, 'recipe': selected['recipe'],
               'selection_exposure': 'already observed development outcomes'})
    for rule in cfg['ensemble']['rules']:
        fuse(root, selected['result'], selected['recipe'], 'fusion_best_'+rule, rule)


def stability(root):
    from component_inspector.research.ksdd2_round2 import train
    _, _, cfg = context(root)
    name, selected = best(root)
    recipe = selected['recipe']
    _, _, metadata, _, _ = source_model(root, recipe['model'])
    write_json(root/'selected-for-stability.json', {'selected_result': name, 'recipe': recipe,
               'selection_exposure': 'observed development; new seeds are repeat-training checks on the same membership'})
    fine = metadata.get('fine', False)
    focal = metadata.get('focal', False)
    lighting = metadata.get('lighting', False)
    prefix = 'fine' if fine else ('combined' if focal and lighting else ('focal' if focal else ('lighting' if lighting else 'plain')))
    outcomes = []
    for seed in cfg['stability_seeds']:
        variant = f'{prefix}_seed{seed}'
        train(root, variant, seed=seed, strategy_override='repeat initialization of selected training configuration')
        if metadata.get('hard_negative_refinement', False):
            refined = f'hard_seed{seed}'
            refine(root, {'model': variant, 'rule': recipe['rule'], 'fusion': None}, refined, seed=seed)
            model = refined
        else:
            model = variant
        scored = score_head(root, model, f'stability_seed{seed}', recipe['rule'], strategy='fixed selected score, new training seed')
        if recipe['fusion']:
            scored = fuse(root, scored, {**recipe, 'model': model, 'fusion': None}, f'stability_seed{seed}_fusion', recipe['fusion'])
        outcomes.append({'seed': seed, 'method': scored['method'], 'metrics': scored['metrics']})
    write_json(root/'stability-summary.json', {'selected_canonical_result': name, 'recipe': recipe,
               'canonical_metrics': selected['result']['metrics'], 'repeat_results': outcomes,
               'claim': 'initialization stability on the same observed split, not independent validation; seeds are not searched for a new headline winner'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['score_heads', 'combined', 'hard', 'fusion_best', 'stability'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    root = args.root.resolve()
    if (root/f'{args.phase}-complete.json').exists():
        raise ValueError('Follow-up already completed')
    if args.phase == 'score_heads':
        from component_inspector.research import ksdd2_round2
        write_json(root/'followup-protocol.json', {
            'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'source_sha256': sha256(Path(__file__)),
            'training_dependency_sha256': sha256(Path(ksdd2_round2.__file__)),
            'fixed_map_rule': 'top_0.1_percent',
            'compound_limit': 1, 'hard_negative_refinement_limit': 1,
            'fusion_rules': context(root)[2]['ensemble']['rules'],
            'repeat_training_seeds': context(root)[2]['stability_seeds'],
            'selection': 'observed development recall subject to FPR strictly below 10%, then fewer false alarms',
            'seed_policy': 'repeat seeds do not enter canonical candidate selection',
            'official_test_access': False})
    elif not (root/'followup-protocol.json').exists():
        raise ValueError('Run score_heads first to record the finite follow-up protocol')
    start = time.perf_counter()
    {'score_heads': score_heads, 'combined': combined, 'hard': hard, 'fusion_best': fusion_best, 'stability': stability}[args.phase](root)
    write_json(root/f'{args.phase}-complete.json', {'elapsed_seconds': time.perf_counter()-start,
               'source_sha256': sha256(Path(__file__)), 'phase': args.phase})


if __name__ == '__main__':
    main()
