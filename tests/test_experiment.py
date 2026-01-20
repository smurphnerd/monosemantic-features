import pytest
import torch
from src.config import SyntheticConfig, ExtractionConfig, ExperimentConfig
from experiments.run_experiment import run_experiment


def test_run_experiment_orthogonal():
    """Full experiment should run and return metrics."""
    config = ExperimentConfig(
        synthetic=SyntheticConfig(
            d=16, n=16, epsilon=0.0, num_representations=20,
            sparsity_mode="fixed", k=2, coef_min_floor=0.3
        ),
        extraction=ExtractionConfig(tau=None, epsilon=0.0),
        seed=42,
        match_threshold=0.9
    )

    result = run_experiment(config)

    assert result.recovery_rate >= 0
    assert result.recovery_rate <= 1
    assert result.mean_alignment >= 0


def test_run_experiment_deterministic():
    """Same seed should give same results."""
    config = ExperimentConfig(
        synthetic=SyntheticConfig(
            d=16, n=16, epsilon=0.0, num_representations=20,
            sparsity_mode="fixed", k=2
        ),
        extraction=ExtractionConfig(tau=None, epsilon=0.0),
        seed=123
    )

    result1 = run_experiment(config)
    result2 = run_experiment(config)

    assert result1.recovery_rate == result2.recovery_rate
