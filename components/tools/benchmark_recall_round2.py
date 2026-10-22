"""Reproduce selected scores from original images and measure CPU latency."""
import argparse
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from component_inspector.research.ksdd2_round1 import bank_score, image_array, rows_for, write_json
from component_inspector.research.ksdd2_round2 import cdf_transform, context, read_json, score_rules
from component_inspector.research.ksdd2_round2_followup import candidate_recipes, source_model
from component_inspector.vision.spike_worker import pretrained_adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--method', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    root = args.root.resolve()
    parent, m, cfg = context(root)
    selected = candidate_recipes(root)[args.method]
    recipe, result = selected['recipe'], selected['result']
    expected = {r['id']: r['score'] for r in result['predictions']}
    start = time.perf_counter()
    adapter = pretrained_adapter('ksdd2_surface', m['config'], args.weights)
    head, state, metadata, _, _ = source_model(root, recipe['model'])
    fine = metadata.get('fine', False)
    generator = torch.Generator().manual_seed(cfg['fine_features']['projection_seed'])
    projection = torch.randn(cfg['fine_features']['projection_channels'], 448, 1, 1, generator=generator)/math.sqrt(448)
    bank = torch.load(parent/'patchcore/bank.pt', weights_only=True) if recipe['fusion'] else None
    normalizers = result['details'].get('normalizers')
    if recipe['fusion'] == 'maximum_normal_cdf' and 'cdf_reference' not in normalizers[0]:
        first_path = parent/'supervised/result.json' if recipe['model'] == 'round1/supervised' else root/recipe['model']/'result.json'
        sources = [read_json(first_path), read_json(parent/'patchcore/result.json')]
        for source, normalization in zip(sources, normalizers):
            values = {r['id']: r['score'] for r in source['calibration_scores']}
            normalization['cdf_reference'] = [values[i] for i in result['details']['normalization_ids']]
    initialization = time.perf_counter()-start
    samples = rows_for(m, 'normal_evaluation')[:8]+rows_for(m, 'positive_evaluation')[:8]
    times, deltas, decisions = [], [], []
    with torch.inference_mode():
        for row in samples:
            tick = time.perf_counter()
            prepared = image_array(row, m['config'])
            features = adapter.model.feature_extractor(adapter._tensor(prepared.rgb_float32))
            f2, f3 = features['layer2'], features['layer3']
            coarse = F.avg_pool2d(torch.cat([f2, F.interpolate(f3, size=f2.shape[-2:], mode='nearest')], 1), 3, stride=1, padding=1).half().float()
            if fine:
                size = features['layer1'].shape[-2:]
                embedding = torch.cat([features['layer1'], F.interpolate(f2, size=size, mode='nearest'), F.interpolate(f3, size=size, mode='nearest')], 1)
                embedding = F.avg_pool2d(F.conv2d(embedding, projection), 3, stride=1, padding=1).half().float()
            else:
                embedding = coarse
            roi = F.interpolate(torch.from_numpy(prepared.valid_roi.astype(np.float32))[None, None], size=embedding.shape[-2:], mode='nearest')[0, 0].bool()
            pixel, image = head((embedding-state['mean'])/state['std'], roi[None])
            value = float(image[0]) if recipe['rule'] == 'image' else score_rules(pixel[0], roi)[recipe['rule']]
            if recipe['fusion']:
                coarse_roi = F.interpolate(torch.from_numpy(prepared.valid_roi.astype(np.float32))[None, None], size=coarse.shape[-2:], mode='nearest')[0, 0].bool()
                second = bank_score(coarse[0].numpy(), coarse_roi.numpy(), bank, m['config']['patchcore']['query_chunk'])
                if recipe['fusion'] == 'maximum_normal_cdf':
                    value = max(float(cdf_transform(normalizers[0]['cdf_reference'], [value])[0]),
                                float(cdf_transform(normalizers[1]['cdf_reference'], [second])[0]))
                else:
                    value = .75*(value-normalizers[0]['median'])/normalizers[0]['iqr']+.25*(second-normalizers[1]['median'])/normalizers[1]['iqr']
            times.append(time.perf_counter()-tick)
            deltas.append(abs(value-expected[row['id']]))
            decisions.append((value > result['threshold_record']['threshold']) == (expected[row['id']] > result['threshold_record']['threshold']))
    if max(deltas) > 1e-4 or not all(decisions):
        raise ValueError('Original-image inference does not match cached evaluation')
    value = {'method': args.method, 'recipe': recipe, 'sample_ids': [r['id'] for r in samples],
             'initialization_seconds': initialization, 'cold_pipeline_seconds': times[0],
             'warm_p95_pipeline_seconds': float(np.percentile(times[1:], 95)), 'pipeline_seconds': times,
             'max_abs_score_difference': max(deltas), 'all_decisions_match': all(decisions),
             'scope': '8 fixed observed-development normals and 8 fixed defects, not chosen by scores; source hash, decode, geometry, frozen backbone, float16 roundtrip, selected head score and optional PatchCore fusion; batch 1 CPU, no UI/network'}
    write_json(root/f'{args.method}-live-benchmark.json', value)
    print(value, flush=True)


if __name__ == '__main__':
    main()
