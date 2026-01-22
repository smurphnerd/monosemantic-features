"""
Integration tests validating the full pipeline on synthetic data.
"""
import pytest
import torch
from src.config import SyntheticConfig, ExtractionConfig, ExperimentConfig
from experiments.run_experiment import run_experiment


class TestOrthogonalRecovery:
    """Test feature recovery with orthogonal (epsilon=0) feature basis."""

    def test_high_recovery_rate(self):
        """Should recover most features with good parameters."""
        config = ExperimentConfig(
            synthetic=SyntheticConfig(
                d=16, n=16, epsilon=0.0, num_representations=100,
                sparsity_mode="fixed", k=2, coef_min_floor=0.3
            ),
            extraction=ExtractionConfig(tau=None, epsilon=0.0),
            seed=42,
            match_threshold=0.9
        )

        result = run_experiment(config)

        # Should recover at least 50% of features (algorithm improves with more representations)
        assert result.recovery_rate >= 0.5, f"Recovery rate {result.recovery_rate} too low"

    def test_high_alignment(self):
        """Recovered features should closely match ground truth."""
        config = ExperimentConfig(
            synthetic=SyntheticConfig(
                d=16, n=16, epsilon=0.0, num_representations=100,
                sparsity_mode="fixed", k=2, coef_min_floor=0.3
            ),
            extraction=ExtractionConfig(tau=None, epsilon=0.0),
            seed=42,
            match_threshold=0.9
        )

        result = run_experiment(config)

        # Matched features should have high alignment
        assert result.mean_alignment > 0.95, f"Mean alignment {result.mean_alignment} too low"

    def test_low_reconstruction_error(self):
        """Should reconstruct representations with low error."""
        config = ExperimentConfig(
            synthetic=SyntheticConfig(
                d=16, n=16, epsilon=0.0, num_representations=100,
                sparsity_mode="fixed", k=2, coef_min_floor=0.3
            ),
            extraction=ExtractionConfig(tau=None, epsilon=0.0),
            seed=42,
        )

        result = run_experiment(config)

        # Reconstruction error should be reasonable
        assert result.reconstruction_error < 1.5, f"Reconstruction error {result.reconstruction_error} too high"

    def test_variable_sparsity(self):
        """Should work with variable sparsity mode."""
        config = ExperimentConfig(
            synthetic=SyntheticConfig(
                d=16, n=16, epsilon=0.0, num_representations=100,
                sparsity_mode="variable", k=3, k_min=2, coef_min_floor=0.3
            ),
            extraction=ExtractionConfig(tau=None, epsilon=0.0),
            seed=42,
        )

        result = run_experiment(config)

        # Should recover features with more representations
        assert result.recovery_rate >= 0.3

    def test_probabilistic_sparsity(self):
        """Should work with probabilistic sparsity mode."""
        config = ExperimentConfig(
            synthetic=SyntheticConfig(
                d=16, n=16, epsilon=0.0, num_representations=100,
                sparsity_mode="probabilistic", k=2, coef_min_floor=0.3
            ),
            extraction=ExtractionConfig(tau=None, epsilon=0.0),
            seed=42,
        )

        result = run_experiment(config)

        # Should recover features with more representations
        assert result.recovery_rate >= 0.3
