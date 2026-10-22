"""Checks against misleading confidence and accidental legacy-head inference."""
import pytest
import torch
from torch import nn

from component_inspector.evaluation.metrics import evaluate
from component_inspector.research.ksdd2_validation import confidence, head_scores, overlap


def counted(tp, fn, fp, tn):
    truth, predictions = {}, []
    for label, flag, count in ((1, True, tp), (1, False, fn), (0, True, fp), (0, False, tn)):
        for _ in range(count):
            identity = str(len(truth))
            truth[identity] = label
            predictions.append({'id': identity, 'status': 'scored', 'score': float(flag), 'flagged': flag})
    return evaluate(predictions, truth)


def test_perfect_small_sample_is_not_confident_confirmation():
    small = confidence(counted(10, 0, 0, 20))
    assert small['point_target_met']
    assert not small['confidence_target_met']
    large = confidence(counted(200, 0, 40, 960))
    assert large['point_target_met'] and large['confidence_target_met']


def test_rejected_images_cannot_disappear_from_confirmation():
    metrics = evaluate([{'id': 'normal', 'status': 'scored', 'score': 0., 'flagged': False}],
                       {'defect': 1, 'normal': 0})
    assert metrics['counts']['FN'] == 1
    with pytest.raises(ValueError, match='complete coverage'):
        confidence(metrics)


def test_validation_ignores_legacy_classifier_and_invalid_padding():
    class SegmentOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.segment = nn.Identity()

        def forward(self, *args):
            raise AssertionError('Legacy classifier must never run')

    head = SegmentOnly()
    state = {'mean': torch.zeros(1, 1, 1, 1), 'std': torch.ones(1, 1, 1, 1)}
    # Padding has the largest logit; it cannot alter the positive defect score.
    features = torch.tensor([[[[2., 6., 9999.]]]])
    roi = torch.tensor([[[True, True, False]]])
    score = head_scores(features, roi, {'test': (head, state, {})})
    assert score['test'] == 6.


def test_exact_decoded_duplicate_detected_despite_different_filenames():
    a = [{'id': 'train/a', 'hashes': {'decoded_sha256': 'same-pixels'}}]
    b = [{'id': 'test/b', 'hashes': {'decoded_sha256': 'same-pixels'}}]
    assert overlap(a, b, 'id') == []
    assert overlap(a, b, 'decoded_sha256') == ['same-pixels']
