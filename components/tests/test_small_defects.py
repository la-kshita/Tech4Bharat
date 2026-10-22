import copy

import numpy as np
from PIL import Image
import pytest
import torch

from component_inspector.research.ksdd2_small_defects import guard, morphology_score, partition, pool, select_method


def test_official_test_and_reserved_ids_rejected_before_image_access():
    row = {'id': 'example', 'recipe': 'ksdd2_surface', 'official_split': 'test', 'label': 1}
    with pytest.raises(ValueError, match='official-training'):
        guard([row])
    row['official_split'] = 'train'
    with pytest.raises(ValueError):
        guard([row], forbidden_ids={'example'})
    with pytest.raises(ValueError):
        guard([row], allowed_ids={'different'})
    guard([row], allowed_ids={'example'})


def test_fine_pool_preserves_small_response_and_excludes_padding():
    raw = torch.zeros(1, 10, 100, requires_grad=True)
    raw.data[0, 0, :3] = 9.
    raw.data[0, -1] = 9999.
    roi = torch.ones_like(raw, dtype=torch.bool); roi[:, -1] = False
    score = pool(raw, roi, 'top3')
    assert float(score.detach()[0]) == 9.
    score.sum().backward()
    assert raw.grad[0, 0, :3].sum() == 1
    assert raw.grad[~roi].sum() == 0


def test_selector_respects_false_alarm_constraint_and_worst_fold_recall():
    def record(recall, fpr):
        return {'metrics': {'recall': {'rate': recall}, 'accepted_normal_fpr': {'rate': fpr}},
                'small_defects': {'detected': 1, 'total': 2}}
    candidates = {'high_false_alarms': [record(1., .11), record(1., .09)],
                  'lucky_one_fold': [record(1., .05), record(.8, .05)],
                  'consistent': [record(.96, .09), record(.95, .09)]}
    before = copy.deepcopy(candidates)
    winner, _ = select_method(candidates)
    assert winner == 'consistent'
    assert candidates == before


def test_classical_small_spot_detector_has_no_label_input():
    cfg = {'gaussian_sigma': .8, 'kernel_sizes': [7, 15, 31], 'contrast_floor': .03}
    normal = np.full((128, 256), 128, np.uint8)
    defective = normal.copy(); defective[61:64, 122:125] = 0
    assert morphology_score(Image.fromarray(normal), cfg) == 0.
    assert morphology_score(Image.fromarray(defective), cfg) > 0.


def test_fold_calibration_is_normal_only_and_outer_originals_excluded():
    by, folds = {}, {}
    for fold in range(3):
        for j, label in enumerate([0, 0, 0, 1]):
            identity = f'{fold}-{j}'
            by[identity] = {'id': identity, 'group': identity, 'recipe': 'ksdd2_surface',
                            'official_split': 'train', 'label': label}
            folds[identity] = fold
    p = {'config': {'seed': 42, 'inner_normal_calibration_count': 1, 'normal_fit_cap': 1},
         'pool_ids': sorted(by), 'folds': folds}
    fitting, calibration = partition(p, by, [0, 1], salt='test')
    assert all(folds[r['id']] != 2 for r in fitting+calibration)
    assert {r['id'] for r in fitting}.isdisjoint(r['id'] for r in calibration)
    assert all(r['label'] == 0 for r in calibration)
    assert sum(r['label'] for r in fitting) == 2
