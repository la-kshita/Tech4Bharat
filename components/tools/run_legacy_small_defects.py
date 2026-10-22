"""Fixed 60+12+12 legacy warm-correction reference on identical outer images."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import psutil
import torch
from torch import nn
from torch.nn import functional as F

from component_inspector.data.acquire import sha256
from component_inspector.evaluation.calibration import calibrate
from component_inspector.research.ksdd2_round1 import DefectHead, digest, image_array, read_json, write_json
from component_inspector.research.ksdd2_small_defects import FeatureStore, context, guard, partition, pool, result, score
from component_inspector.vision.resources import run_bounded


CFG = {'family': 'legacy_supervised', 'grid': 'coarse', 'channels': 384, 'hidden': 32,
       'pool': 'top_0.1_percent', 'base_epochs': 60, 'hard_refinement_epochs': 12,
       'aligned_continuation_epochs': 12, 'base_learning_rate': .001,
       'refinement_learning_rate': .0002, 'weight_decay': .0001,
       'seed': 4217, 'normalization': 'original 128 normals plus outer-training positives; whole coarse canvas; float32 moments',
       'calibration_levels': [.08, .10], 'selection_role': 'fixed reference only; never selects candidate'}


class LegacyMap(nn.Module):
    def __init__(self, head):
        super().__init__(); self.head = head

    def forward(self, x):
        return self.head.segment(x)[:, 0]


def legacy_scorer(state):
    head = DefectHead(384, 32); head.load_state_dict(state['head']); head.eval()
    return LegacyMap(head).eval()


def fit(head, rows, store, mean, std, *, epochs, lr, aligned):
    optimizer = torch.optim.AdamW(head.segment.parameters() if aligned else head.parameters(),
                                  lr=lr, weight_decay=.0001)
    generator = torch.Generator().manual_seed(4217)
    history = []
    for epoch in range(epochs):
        head.train(); total = 0.
        for indices in torch.randperm(len(rows), generator=generator).split(4):
            batch = [rows[int(i)] for i in indices]
            features = torch.from_numpy(np.stack([store.feature(r['id'], CFG) for r in batch]))
            roi = torch.from_numpy(np.stack([store.roi(r['id'], CFG) for r in batch]))
            target = torch.from_numpy(np.stack([store.target(r['id'], CFG) if r['label'] else np.zeros(roi.shape[-2:], np.float32) for r in batch]))
            labels = torch.tensor([float(r['label']) for r in batch])
            raw, image = head((features-mean)/std, roi)
            bce = F.binary_cross_entropy_with_logits(raw, target, pos_weight=torch.tensor(8.), reduction='none')
            probability = raw.sigmoid(); pt = probability*target+(1-probability)*(1-target)
            pixel = (bce*(1-pt).square()*roi).sum()/roi.sum()
            image = pool(raw, roi, CFG['pool']) if aligned else image
            loss = pixel+F.binary_cross_entropy_with_logits(image, labels)
            if not torch.isfinite(loss): raise ValueError('Nonfinite legacy loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(head.segment.parameters() if aligned else head.parameters(), 5.)
            optimizer.step(); total += float(loss.detach())*len(batch)
        history.append({'epoch': epoch+1, 'loss': total/len(rows)})
        if epoch % 5 == 0 or epoch+1 == epochs:
            print('LEGACY', 'aligned' if aligned else 'learned_image', epochs, epoch+1, history[-1]['loss'], flush=True)
    return history


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=['register', 'run', 'fit'], required=True)
    parser.add_argument('--fold', type=int, choices=[0, 1, 2]); args = parser.parse_args()
    root = args.root.resolve(); p, m, by = context(root)
    if args.phase == 'register':
        if any(root.glob('outer-*/eval-*.json')): raise ValueError('Register reference before outer results')
        record = {'protocol_identity': p['identity'], 'configuration': CFG,
                  'source_sha256': sha256(Path(__file__)),
                  'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  'registered_before_outer_results': True,
                  'purpose': 'full prior warm-corrected training recipe, fresh fold-specific fit; fixed secondary reference, not a new selectable method',
                  'comparison': 'common .08 calibration plus original .10 policy; both frozen before outer results',
                  'no_official_test_access': True}
        record['identity'] = digest(record); write_json(root/'legacy-reference-plan.json', record)
        print('REGISTERED LEGACY REFERENCE', record['identity'], flush=True); return
    registered = read_json(root/'legacy-reference-plan.json')
    assert sha256(Path(__file__)) == registered['source_sha256']
    if args.phase == 'run':
        if not (root/'nested-results.json').exists(): raise ValueError('Wait for main study')
        for fold in range(3):
            resource_path = root/f'legacy-outer-{fold}-resources.json'
            if resource_path.exists():
                if read_json(resource_path)['status'] != 'completed': raise RuntimeError('Preserved failure')
                continue
            cap = min(4*1024**3, int(.60*psutil.virtual_memory().available))
            print('START LEGACY REFERENCE', fold, flush=True)
            resources = run_bounded([sys.executable, str(Path(__file__).resolve()), '--root', str(root),
                                     '--phase', 'fit', '--fold', str(fold)], root/f'legacy-outer-{fold}-worker.log',
                                     max_rss_bytes=cap, max_seconds=1800)
            write_json(resource_path, resources); print('END LEGACY REFERENCE', fold, resources, flush=True)
            if resources['status'] != 'completed': raise RuntimeError('Legacy reference failed')
        return
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    fold = args.fold; directory = root/f'legacy-outer-{fold}'; directory.mkdir(exist_ok=False)
    common_fitting, calibration_rows = partition(p, by, [f for f in range(3) if f != fold], salt=f'outer-{fold}')
    normals = [r for r in common_fitting if not r['label']][:128]
    base = sorted(normals, key=lambda r: r['id'])+sorted([r for r in common_fitting if r['label']], key=lambda r: r['id'])
    calibration_ids = {r['id'] for r in calibration_rows}
    forbidden = {i for i in p['pool_ids'] if p['folds'][i] == fold} | set(p['reserved_normal_calibration_ids'])
    guard(base+calibration_rows, forbidden_ids=forbidden)
    write_json(directory/'plan.json', {'method': 'legacy_warm_recipe', 'configuration': CFG, 'fold': fold,
               'base_fitting_ids': [r['id'] for r in base], 'calibration_ids': [r['id'] for r in calibration_rows],
               'reference_identity': registered['identity'], 'excluded_outer_ids': sorted(i for i in p['pool_ids'] if p['folds'][i] == fold)})
    store = FeatureStore(root, p); sums, squares = torch.zeros(384), torch.zeros(384); count = 0
    for row in base:
        x = torch.from_numpy(store.feature(row['id'], CFG))
        sums += x.sum((1, 2)); squares += x.square().sum((1, 2)); count += x.shape[1]*x.shape[2]
    mean = (sums/count).reshape(1, -1, 1, 1)
    std = (squares/count-(sums/count).square()).clamp_min(.01).sqrt().reshape(1, -1, 1, 1)
    torch.manual_seed(4217); head = DefectHead(384, 32); start = time.perf_counter()
    base_history = fit(head, base, store, mean, std, epochs=60, lr=.001, aligned=False)
    head.eval(); map_head = LegacyMap(head).eval(); state = {'mean': mean, 'std': std}
    normal_base = {r['id'] for r in normals}
    candidates = [by[i] for i in p['pool_ids'] if p['folds'][i] != fold and not by[i]['label']
                  and i not in normal_base | calibration_ids]
    ranked = [(score(store, r['id'], CFG, state, map_head), r) for r in candidates]
    hard = [r for _, r in sorted(ranked, key=lambda pair: (-pair[0], pair[1]['id']))[:128]]
    guard(hard, forbidden_ids=forbidden | calibration_ids | normal_base)
    for row in hard:
        if image_array(row, m['config'], mask=True).any(): raise ValueError('Hard normal has defect mask')
    fitting = base+hard
    refinement_history = fit(head, fitting, store, mean, std, epochs=12, lr=.0002, aligned=False)
    head.classify.requires_grad_(False)
    aligned_history = fit(head, fitting, store, mean, std, epochs=12, lr=.0002, aligned=True)
    torch.save({'head': head.state_dict(), 'mean': mean, 'std': std}, directory/'model.pt')
    write_json(directory/'training.json', {'base_fitting_ids': [r['id'] for r in base],
               'fitting_ids': [r['id'] for r in fitting], 'hard_normal_ids': [r['id'] for r in hard],
               'mining_candidate_ids': [r['id'] for r in candidates], 'normalization_ids': [r['id'] for r in base],
               'base_history': base_history, 'hard_refinement_history': refinement_history,
               'aligned_continuation_history': aligned_history, 'fit_seconds': time.perf_counter()-start,
               'seed': 4217, 'model_sha256': sha256(directory/'model.pt')})
    state = torch.load(directory/'model.pt', weights_only=True); map_head = legacy_scorer(state)
    calibration = [{'id': r['id'], 'score': score(store, r['id'], CFG, state, map_head)} for r in calibration_rows]
    rows = [by[i] for i in p['pool_ids'] if p['folds'][i] == fold]
    values = {r['id']: score(store, r['id'], CFG, state, map_head) for r in rows}
    for alpha in CFG['calibration_levels']:
        threshold = calibrate([r['score'] for r in calibration], alpha=alpha)
        result(directory, f'eval-{fold}-alpha{alpha:.2f}.json', rows, values, calibration, threshold, p)
    write_json(directory/'complete.json', {'model_sha256': sha256(directory/'model.pt'),
               'reference_identity': registered['identity'], 'protocol_identity': p['identity']})


if __name__ == '__main__': main()
