"""Small mathematical checks for round 2 scoring, loss and calibration fusion."""
import numpy as np
import pytest
import torch
import json
from torch.nn import functional as F

from component_inspector.research.ksdd2_round2 import score_rules, focal_loss, cdf_transform


def test_aggregation_excludes_padding_and_normalizes_local_edge_windows():
    values = torch.full((5, 5), 2.)
    roi = torch.zeros(5, 5, dtype=torch.bool)
    roi[1:4, 1:4] = True
    values[~roi] = 999.
    scores = score_rules(values, roi)
    assert all(v == pytest.approx(2.) for v in scores.values())
    with pytest.raises(ValueError, match='Invalid scoring map'):
        score_rules(values, torch.zeros_like(roi))


def test_focal_gamma_zero_matches_parent_weighted_bce_and_has_finite_gradients():
    x = torch.tensor([-2., 0., 3.], requires_grad=True)
    y = torch.tensor([0., 1., 1.])
    expected = F.binary_cross_entropy_with_logits(x, y, pos_weight=torch.tensor(8.), reduction='none')
    assert torch.allclose(focal_loss(x, y, gamma=0), expected)
    loss = focal_loss(x, y, gamma=2)
    assert (loss <= expected).all()
    loss.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_normal_cdf_uses_reference_only_and_handles_ties_monotonically():
    out = cdf_transform([1., 2., 2., 4.], [-1., 1., 2., 3., 4., 99.])
    assert np.allclose(out, [0., .2, .6, .6, .8, .8])
    assert (np.diff(out) >= 0).all()


@pytest.mark.parametrize('variant', ['fine', 'combined', 'plain_seed4218'])
def test_training_variants_stream_correct_features_and_exclude_reserved_items(tmp_path, monkeypatch, variant):
    from component_inspector.research import ksdd2_round2 as r2
    torch.set_num_threads(1)
    parent = tmp_path/'parent'
    root = tmp_path/'new'
    root.mkdir()
    roles = [('n0', 'normal_fit', 0), ('n1', 'normal_fit', 0),
             ('p0', 'positive_train', 1), ('p1', 'positive_train', 1),
             ('e0', 'normal_evaluation', 0), ('e1', 'positive_evaluation', 1)]
    roles += [(f'c{i}', 'normal_calibration', 0) for i in range(9)]
    records = [{'id': i, 'v2_role': role, 'label': label, 'group': i,
                'recipe': 'ksdd2_surface', 'official_split': 'train', 'quarantine_reason': None}
               for i, role, label in roles]
    manifest = {'records': records, 'original_fit_ids': ['n0', 'n1'], 'identity': 'fixture',
                'config': {'supervised': {'hidden_channels': 3, 'epochs': 2, 'batch_size': 2,
                           'learning_rate': .001, 'weight_decay': .0001, 'pixel_positive_weight': 8.}}}
    config = {'seed': 4217, 'lighting': {'training_probability': .5},
              'focal': {'positive_weight': 8., 'gamma': 2.}}
    rng = np.random.default_rng(1)
    for directory, channels in [(parent/'features', 4), (root/'fine_features', 7), (root/'lighting_features', 4)]:
        directory.mkdir(parents=True)
        index = {}
        for i, role, label in roles:
            if directory.name == 'lighting_features' and role not in {'normal_fit', 'positive_train'}:
                continue
            np.save(directory/f'{i}.npy', rng.normal(size=(channels, 4, 4)).astype(np.float16))
            np.save(directory/f'{i}-roi.npy', np.ones((4, 4), bool))
            if role == 'positive_train':
                target = np.zeros((4, 4), np.float32)
                target[1, 1] = 1.
                np.save(directory/f'{i}-target.npy', target)
            index[i] = {'key': i, 'role': role}
        (directory/'index.json').write_text(json.dumps({'shape': [channels, 4, 4], 'rows': index}))
    monkeypatch.setattr(r2, 'context', lambda path: (parent, manifest, config))
    r2.train(root, variant, seed=4218, strategy_override='software fixture')
    value = r2.read_json(root/variant/'result.json')
    metadata = r2.read_json(root/variant/'model.json')
    assert set(value['details']['fitting_ids']) == {'n0', 'n1', 'p0', 'p1'}
    assert {r['id'] for r in value['predictions']} == {'e0', 'e1'}
    assert metadata['channels'] == (7 if variant == 'fine' else 4)
    assert metadata['fine'] == (variant == 'fine')
    assert metadata['focal'] == metadata['lighting'] == (variant == 'combined')
    assert metadata['seed'] == 4218
    assert value['details']['strategy'] == 'software fixture'
