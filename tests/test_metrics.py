import pytest
import torch
from src.metrics import match_features


def test_match_features_perfect_recovery():
    """Identical features should match perfectly."""
    ground_truth = torch.eye(4)
    extracted = torch.eye(4)

    matching, scores = match_features(extracted, ground_truth, threshold=0.9)

    assert len(matching) == 4
    assert torch.all(scores > 0.99)


def test_match_features_partial_recovery():
    """Should only match features above threshold."""
    ground_truth = torch.eye(4)
    extracted = torch.eye(4)[:2]  # Only first 2 features

    matching, scores = match_features(extracted, ground_truth, threshold=0.9)

    assert len(matching) == 2


def test_match_features_sign_invariant():
    """Matching should be invariant to sign flips."""
    ground_truth = torch.eye(4)
    extracted = -torch.eye(4)  # Negated

    matching, scores = match_features(extracted, ground_truth, threshold=0.9)

    assert len(matching) == 4
    assert torch.all(scores > 0.99)
