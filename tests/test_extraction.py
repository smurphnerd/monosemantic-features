import pytest
import torch
from src.config import SyntheticConfig, ExtractionConfig, compute_coef_min
from src.extraction import compute_tau_bounds, resolve_tau


def test_compute_tau_bounds_orthogonal():
    """For orthogonal features, tau bounds should be well-separated."""
    k = 3
    epsilon = 0.0
    coef_min = 0.1
    coef_max = 1.0

    tau_min, tau_max = compute_tau_bounds(k, epsilon, coef_min, coef_max)

    # tau_max should be less than tau_min for valid separation
    assert tau_max < tau_min


def test_resolve_tau_auto():
    """When tau is None, it should be auto-derived."""
    synthetic = SyntheticConfig(epsilon=0.0, k=3, coef_min_floor=0.1)
    extraction = ExtractionConfig(tau=None, tau_margin=0.5)

    tau = resolve_tau(extraction, synthetic)

    assert tau > 0
    assert tau < 1


def test_resolve_tau_manual():
    """When tau is set, it should be used directly."""
    synthetic = SyntheticConfig(epsilon=0.0, k=3)
    extraction = ExtractionConfig(tau=0.7)

    tau = resolve_tau(extraction, synthetic)

    assert tau == 0.7
