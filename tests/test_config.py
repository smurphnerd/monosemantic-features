import pytest
from src.config import SyntheticConfig, compute_coef_min


def test_synthetic_config_defaults():
    config = SyntheticConfig()
    assert config.d == 64
    assert config.n == 64
    assert config.epsilon == 0.0
    assert config.num_representations == 1000
    assert config.sparsity_mode == "fixed"
    assert config.k == 3


def test_compute_coef_min_with_epsilon_zero():
    config = SyntheticConfig(epsilon=0.0, k=3, coef_factor=10.0, coef_min_floor=0.1)
    # When epsilon=0, coef_min should be coef_min_floor
    assert compute_coef_min(config) == 0.1


def test_compute_coef_min_with_epsilon_nonzero():
    config = SyntheticConfig(epsilon=0.1, k=3, coef_factor=10.0, coef_min_floor=0.1)
    # coef_min = max(10 * 3 * 0.1, 0.1) = max(3.0, 0.1) = 3.0
    assert compute_coef_min(config) == 3.0
