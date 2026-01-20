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


from src.metrics import evaluate, MetricsResult


def test_evaluate_perfect_recovery():
    """Perfect recovery should give recovery_rate=1.0."""
    ground_truth = torch.eye(4)
    extracted = torch.eye(4)
    representations = torch.eye(4)  # Each representation is one feature
    coefficients = torch.eye(4)  # Each has coefficient 1 on one feature

    result = evaluate(extracted, ground_truth, representations, coefficients)

    assert isinstance(result, MetricsResult)
    assert result.recovery_rate == 1.0
    assert result.mean_alignment > 0.99


def test_evaluate_partial_recovery():
    """Partial recovery should give recovery_rate < 1.0."""
    ground_truth = torch.eye(4)
    extracted = torch.eye(4)[:2]  # Only 2 features recovered
    representations = torch.eye(4)
    coefficients = torch.eye(4)

    result = evaluate(extracted, ground_truth, representations, coefficients)

    assert result.recovery_rate == 0.5  # 2 out of 4


def test_evaluate_returns_diagnostics():
    """Evaluation should return matching diagnostics."""
    ground_truth = torch.eye(4)
    extracted = torch.eye(4)[:3]
    representations = torch.eye(4)
    coefficients = torch.eye(4)

    result = evaluate(extracted, ground_truth, representations, coefficients)

    assert len(result.feature_matching) == 3
    assert len(result.unmatched_true) == 1
    assert len(result.unmatched_extracted) == 0
