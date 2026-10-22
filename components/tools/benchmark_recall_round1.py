"""Check real decode/backbone/score latency and parity for a fitted candidate."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from component_inspector.research.ksdd2_round1 import (
    DefectHead, bank_score, image_array, load_manifest, rows_for, write_json,
)
from component_inspector.vision.spike_worker import pretrained_adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--method', choices=['supervised', 'patchcore', 'coverage128', 'coverage256'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    root = args.root.resolve()
    m = load_manifest(root)
    result = json.loads((root/args.method/'result.json').read_text())
    scores = {r['id']: r['score'] for r in result['predictions']}
    init_tick = time.perf_counter()
    adapter = pretrained_adapter('ksdd2_surface', m['config'], args.weights)
    if args.method == 'supervised':
        state = torch.load(root/'supervised/head.pt', weights_only=True)
        head = DefectHead(384, m['config']['supervised']['hidden_channels'])
        head.load_state_dict(state['head'])
        head.eval()
    else:
        bank = torch.load(root/args.method/'bank.pt', weights_only=True)
    initialization = time.perf_counter()-init_tick
    times, deltas = [], []
    # Fixed ID order from development normals; no labels choose these examples.
    samples = rows_for(m, 'normal_evaluation')[:16]
    with torch.inference_mode():
        for row in samples:
            tick = time.perf_counter()
            prepared = image_array(row, m['config'])
            features = adapter.model.feature_extractor(adapter._tensor(prepared.rgb_float32))
            f2, f3 = features['layer2'], features['layer3']
            embedding = torch.cat([f2, F.interpolate(f3, size=f2.shape[-2:], mode='nearest')], dim=1)
            embedding = F.avg_pool2d(embedding, 3, stride=1, padding=1)
            # The fitted CPU recipe explicitly uses half-precision disk caches.
            embedding = embedding.half().float()
            roi = F.interpolate(torch.from_numpy(prepared.valid_roi.astype(np.float32))[None, None],
                                size=embedding.shape[-2:], mode='nearest')[0, 0].bool()
            if args.method == 'supervised':
                _, image = head((embedding-state['mean'])/state['std'], roi[None])
                score = float(image[0])
            else:
                score = bank_score(embedding[0].numpy(), roi.numpy(), bank, m['config']['patchcore']['query_chunk'])
            times.append(time.perf_counter()-tick)
            deltas.append(abs(score-scores[row['id']]))
    if max(deltas) > 1e-4:
        raise ValueError('Actual inference does not match cached evaluation recipe')
    value = {'method': args.method, 'sample_ids': [r['id'] for r in samples],
             'initialization_seconds': initialization, 'cold_pipeline_seconds': times[0],
             'warm_p95_pipeline_seconds': float(np.percentile(times[1:], 95)),
             'pipeline_seconds': times, 'max_abs_score_difference': max(deltas),
             'scope': '16 fixed development normals; source hash/decode/geometry/backbone/quantization/score, batch 1 CPU; no UI/network'}
    write_json(root/f'{args.method}-live-benchmark.json', value)
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
