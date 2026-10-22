"""Research boundaries and model primitives on fictional, hand-computable data."""
import copy
import json
import numpy as np
import pytest
import torch

from component_inspector.research.ksdd2_round1 import (
    guard, farthest_points, bank_score, DefectHead, image_array, gaussian_by_position, supervised,
)


def row(role='normal_fit', label=0):
    return {'recipe': 'ksdd2_surface', 'official_split': 'train',
            'quarantine_reason': None, 'label': label, 'group': 'a', 'v2_role': role}


def test_research_rejects_test_and_cross_role_groups_before_any_decode():
    r = row()
    guard([r])
    wrong = copy.deepcopy(r)
    wrong['official_split'] = 'test'
    with pytest.raises(ValueError, match='training originals'):
        image_array(wrong, {})
    wrong = row('normal_calibration')
    with pytest.raises(ValueError, match='crosses roles'):
        guard([r, wrong])
    with pytest.raises(ValueError, match='normal-only'):
        guard([row('normal_fit', 1)])


def test_diversity_selection_covers_extremes_without_duplicates_or_labels():
    x = np.array([[0., 0.], [1., 0.], [10., 0.], [11., 0.]], np.float32)
    chosen = farthest_points(x, 3)
    assert len(set(chosen)) == 3
    assert {0, 3}.issubset(set(chosen))
    assert chosen == farthest_points(x, 3)
    # Even all-identical vectors cannot yield duplicate indices.
    assert len(set(farthest_points(np.zeros((5, 2), np.float32), 5))) == 5


def test_patchcore_score_ignores_padding_and_matches_direct_nearest_distance():
    feature = np.array([[[0., 999.], [3., 999.]], [[0., 999.], [4., 999.]]], np.float32)
    roi = np.array([[True, False], [True, False]])
    bank = torch.tensor([[0., 0.], [1., 0.]])
    assert bank_score(feature, roi, bank, chunk=1) == pytest.approx(np.sqrt(20))
    with pytest.raises(ValueError, match='Invalid query'):
        bank_score(feature, np.zeros((2, 2), bool), bank)


def test_supervised_head_gradients_and_tensor_artifact_prediction_parity(tmp_path):
    torch.manual_seed(3)
    head = DefectHead(4, 3)
    x = torch.rand(2, 4, 8, 8)
    roi = torch.ones(2, 8, 8, dtype=torch.bool)
    pixel, image = head(x, roi)
    assert pixel.shape == (2, 8, 8) and image.shape == (2,)
    loss = pixel.square().mean()+image.square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    path = tmp_path/'head.pt'
    torch.save(head.state_dict(), path)
    restored = DefectHead(4, 3)
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert torch.equal(head(x, roi)[1], restored(x, roi)[1])


def test_chunked_gaussian_matches_pinned_library_fit_equations():
    from anomalib.models.components.stats.multi_variate_gaussian import MultiVariateGaussian
    torch.manual_seed(9)
    embedding = torch.randn(8, 4, 3, 5)
    expected_mean, expected_inverse = MultiVariateGaussian().fit(embedding)
    mean, inverse = gaussian_by_position(embedding.reshape(8, 4, 15).numpy(), chunk=4)
    assert torch.allclose(mean, expected_mean, atol=1e-6, rtol=1e-6)
    assert torch.allclose(inverse, expected_inverse, atol=1e-6, rtol=1e-6)


def test_streamed_supervised_fit_keeps_evaluation_and_calibration_out_of_training(tmp_path):
    """A real two-epoch software fixture exercises disk batches and result joins."""
    directory = tmp_path/'features'
    directory.mkdir()
    roles = [('fit0', 'normal_fit', 0), ('fit1', 'normal_fit', 0),
             ('train0', 'positive_train', 1), ('train1', 'positive_train', 1),
             ('eval0', 'normal_evaluation', 0), ('eval1', 'normal_evaluation', 0),
             ('eval2', 'positive_evaluation', 1)]
    roles += [(f'cal{i}', 'normal_calibration', 0) for i in range(9)]
    records, index = [], {}
    rng = np.random.default_rng(10)
    for identity, role, label in roles:
        records.append({'id': identity, 'label': label, 'group': identity, 'v2_role': role})
        np.save(directory/f'{identity}.npy', rng.normal(size=(384, 4, 4)).astype(np.float16))
        np.save(directory/f'{identity}-roi.npy', np.ones((4, 4), bool))
        index[identity] = {'key': identity, 'role': role, 'extract_seconds': 0.0}
        if role == 'positive_train':
            target = np.zeros((4, 4), np.float32)
            target[1, 1] = 1
            np.save(directory/f'{identity}-target.npy', target)
    (directory/'index.json').write_text(json.dumps({'shape': [384, 4, 4], 'rows': index}))
    m = {'records': records, 'original_fit_ids': ['fit0', 'fit1'], 'identity': 'fictional',
         'claim': 'fictional software fixture', 'config': {'seed': 4217, 'normal_calibration_alpha': .1,
         'supervised': {'epochs': 2, 'batch_size': 2, 'hidden_channels': 3, 'learning_rate': .001,
                        'weight_decay': .0001, 'pixel_positive_weight': 8., 'image_loss_weight': 1.}}}
    supervised(tmp_path, m)
    result = json.loads((tmp_path/'supervised/result.json').read_text())
    assert set(result['training_ids']) == {'fit0', 'fit1', 'train0', 'train1'}
    assert {r['id'] for r in result['predictions']} == {'eval0', 'eval1', 'eval2'}
    assert len(result['calibration_scores']) == 9
    assert len(result['training_history']) == 2
