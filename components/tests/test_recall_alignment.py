"""Regression at the objective seam: classifier sign cannot reverse learning."""
import torch

from component_inspector.research.ksdd2_round1 import DefectHead
from component_inspector.research.ksdd2_aligned import aligned_objective, fixed_pool


def test_segmentation_learning_is_independent_of_unused_classifier_sign():
    torch.manual_seed(12)
    head = DefectHead(4, 3)
    features = torch.rand(2, 4, 4, 4)
    roi = torch.ones(2, 4, 4, dtype=torch.bool)
    target = torch.zeros(2, 4, 4)
    target[1, 1, 1] = 1.
    labels = torch.tensor([0., 1.])
    losses, gradients = [], []
    for sign in [-1., 1.]:
        with torch.no_grad():
            head.classify.weight.fill_(sign)
            head.classify.bias.zero_()
        head.zero_grad(set_to_none=True)
        loss = aligned_objective(head, features, roi, target, labels)
        loss.backward()
        losses.append(loss.detach().clone())
        gradients.append(torch.cat([p.grad.flatten() for p in head.segment.parameters()]))
    assert torch.equal(losses[0], losses[1])
    assert torch.equal(gradients[0], gradients[1])


def test_positive_image_loss_pushes_valid_peak_up_and_ignores_padding():
    raw = torch.tensor([[[999., -1.], [999., 2.]]], requires_grad=True)
    roi = torch.tensor([[[False, True], [False, True]]])
    score = fixed_pool(raw, roi)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(score, torch.ones(1))
    loss.backward()
    assert score.item() == 2.
    assert (raw.grad[~roi] == 0).all()
    assert raw.grad[0, 1, 1] < 0  # Gradient descent increases the positive score.


def test_differentiable_training_pool_matches_existing_inference_pool():
    from component_inspector.research.ksdd2_round2 import score_rules
    torch.manual_seed(7)
    raw = torch.randn(3, 32, 80)
    roi = torch.rand(3, 32, 80) > .1
    observed = fixed_pool(raw, roi)
    expected = torch.tensor([score_rules(r, mask)['top_0.1_percent'] for r, mask in zip(raw, roi)])
    assert torch.equal(observed, expected)
