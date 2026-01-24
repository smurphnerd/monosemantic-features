import pytest
from src.config import SyntheticConfig, compute_coef_min


def test_synthetic_config_defaults():
    config = SyntheticConfig()
    assert config.d == 64
    assert config.n == 64
    assert config.num_representations == 1000
    assert config.sparsity_mode == "fixed"
    assert config.k == 3


def test_compute_coef_min_with_epsilon_zero():
    config = SyntheticConfig()
    result = compute_coef_min(config, epsilon=0.0)
    assert result == config.coef_min_floor


def test_compute_coef_min_with_epsilon_nonzero():
    config = SyntheticConfig(coef_factor=10.0, k=3, coef_min_floor=0.1)
    result = compute_coef_min(config, epsilon=0.05)
    # 10 * 3 * 0.05 = 1.5, which is > 0.1
    assert result == 1.5
