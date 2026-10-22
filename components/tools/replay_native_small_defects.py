"""Recompute every nested/control/repeat prediction from original training pixels."""
import argparse
from pathlib import Path
import sys

import numpy as np
import psutil
import torch
from torch.nn import functional as F

from component_inspector.data.acquire import sha256
from component_inspector.research.ksdd2_round1 import image_array, read_json, write_json
from component_inspector.research.ksdd2_small_defects import (
    NativeStore, context, morphology_score, score, scorer,
)
from component_inspector.vision.resources import run_bounded
from component_inspector.vision.spike_worker import pretrained_adapter
from run_legacy_small_defects import legacy_scorer


@torch.inference_mode()
def replay(root):
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    p, manifest, by = context(root)
    selected = read_json(root/'selection.json')
    models, expected, model_files = {}, {}, {}
    for fold in range(3):
        for role, directory in [
                ('primary', root/f'outer-{fold}'), ('control', root/f'control-outer-{fold}'),
                ('repeat1000', root/f'repeat-1000-outer-{fold}'),
                ('repeat2000', root/f'repeat-2000-outer-{fold}'),
                ('legacy08', root/f'legacy-outer-{fold}'), ('legacy10', root/f'legacy-outer-{fold}')]:
            plan = read_json(directory/'plan.json')
            method = plan['method']
            cfg = plan['configuration'] if role.startswith('legacy') else p['config']['methods'][method]
            model_hash = sha256(directory/'model.pt')
            assert model_hash == read_json(directory/'training.json')['model_sha256']
            state = torch.load(directory/'model.pt', weights_only=True, map_location='cpu')
            suffix = '-alpha0.08' if role == 'legacy08' else '-alpha0.10' if role == 'legacy10' else ''
            result = read_json(directory/f'eval-{fold}{suffix}.json')
            key = (fold, role)
            model_head = legacy_scorer(state) if role.startswith('legacy') else scorer(state, cfg)
            models[key] = (cfg, state, model_head, result['threshold_record']['threshold'])
            expected[key] = {r['id']: r for r in result['predictions']}
            model_files[str(directory/'model.pt')] = model_hash
    adapter = pretrained_adapter('ksdd2_surface', manifest['config'], Path(p['weights']))
    roles = ['primary', 'control', 'repeat1000', 'repeat2000', 'legacy08', 'legacy10']
    differences = {role: 0. for role in roles}; mismatches = {role: 0 for role in roles}
    native_predictions = []
    # Reverse order relative to the cache-build/source-ID ordering.
    for i, identity in enumerate(reversed(p['pool_ids'])):
        row = by[identity]; fold = p['folds'][identity]
        prepared = image_array(row, manifest['config'])
        tensors = adapter.model.feature_extractor(adapter._tensor(prepared.rgb_float32))
        f1, f2, f3 = (tensors[f'layer{n}'] for n in (1, 2, 3))
        coarse = F.avg_pool2d(torch.cat([f2, F.interpolate(f3, size=f2.shape[-2:], mode='nearest')], 1), 3, stride=1, padding=1)[0].half().float().numpy()
        fine = torch.cat([f1, F.interpolate(f2, size=f1.shape[-2:], mode='nearest'),
                          F.interpolate(f3, size=f1.shape[-2:], mode='nearest')], 1)[0].half().float().numpy()
        valid = torch.from_numpy(prepared.valid_roi.astype(np.float32))[None, None]
        coarse_roi = F.interpolate(valid, size=coarse.shape[-2:], mode='nearest')[0, 0].bool().numpy()
        fine_roi = F.interpolate(valid, size=fine.shape[-2:], mode='nearest')[0, 0].bool().numpy()
        morphology = 0.
        if any(models[(fold, role)][0]['family'] == 'morphology' for role in roles):
            from PIL import Image
            with Image.open(row['path']) as image:
                morphology = morphology_score(image, p['config']['methods']['morphology'])
        store = NativeStore(identity, coarse, fine, coarse_roi, fine_roi, morphology)
        out = {'id': identity, 'outer_fold': fold, 'predictions': {}}
        for role in roles:
            cfg, state, head, threshold = models[(fold, role)]
            value = score(store, identity, cfg, state, head)
            flag = value > threshold
            prior = expected[(fold, role)][identity]
            differences[role] = max(differences[role], abs(value-prior['score']))
            mismatches[role] += int(flag != prior['flagged'])
            out['predictions'][role] = {'score': value, 'flagged': flag}
        native_predictions.append(out)
        if i % 100 == 0 or i+1 == len(p['pool_ids']):
            print('FULL NATIVE REPLAY', i+1, '/', len(p['pool_ids']), flush=True)
    for file, expected_hash in model_files.items(): assert sha256(file) == expected_hash
    record = {'protocol_identity': p['identity'], 'selection_identity': selected['identity'],
              'source_sha256': sha256(Path(__file__)), 'image_count': len(native_predictions),
              'model_evaluations': len(native_predictions)*len(roles),
              'max_absolute_score_difference': differences, 'decision_mismatches': mismatches,
              'all_decisions_match': not any(mismatches.values()),
              'all_scores_bitwise_equal': not any(differences.values()),
              'all_scores_within_1e_5': max(differences.values()) <= 1e-5,
              'model_files_unchanged': model_files, 'official_test_accessed': False,
              'scope': 'all nested/control/refit/full-legacy decisions regenerated from original training RGB images; legacy has two fixed threshold policies on the same weights; no cached feature arrays read',
              'repeated_observations_do_not_add_independent_images': True}
    write_json(root/'full-native-replay.json', record)
    write_json(root/'full-native-predictions.json', {'protocol_identity': p['identity'], 'rows': native_predictions})
    if not record['all_decisions_match'] or not record['all_scores_within_1e_5']:
        raise ValueError('Full original-image replay does not match saved evaluation')
    print('FULL NATIVE REPLAY PASSED', differences, mismatches, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args(); root = args.root.resolve()
    if args.worker:
        replay(root); return
    cap = min(4*1024**3, int(.60*psutil.virtual_memory().available))
    resource = run_bounded([sys.executable, str(Path(__file__).resolve()), '--root', str(root), '--worker'],
                           root/'full-native-worker.log', max_rss_bytes=cap, max_seconds=1800)
    write_json(root/'full-native-resources.json', resource)
    print(resource, flush=True)
    if resource['status'] != 'completed': raise RuntimeError('Preserve native replay failure')


if __name__ == '__main__': main()
