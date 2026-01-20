import pytest
import torch
from src.config import SyntheticConfig, ExtractionConfig, compute_coef_min
from src.extraction import compute_tau_bounds, resolve_tau, find_neighbors, cluster_by_neighbors


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


def test_find_neighbors_includes_self():
    """A representation should always be its own neighbor."""
    representations = torch.eye(4)  # 4 orthogonal unit vectors
    neighbors = find_neighbors(representations, target_idx=0, tau=0.5)
    assert 0 in neighbors


def test_find_neighbors_orthogonal():
    """Orthogonal representations shouldn't be neighbors (except self)."""
    representations = torch.eye(4)
    neighbors = find_neighbors(representations, target_idx=0, tau=0.5)
    assert len(neighbors) == 1
    assert neighbors[0] == 0


def test_find_neighbors_similar():
    """Similar representations should be neighbors."""
    representations = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],  # Similar to first
        [0.0, 1.0, 0.0],  # Orthogonal to first
    ])
    # Normalize
    representations = representations / representations.norm(dim=1, keepdim=True)

    neighbors = find_neighbors(representations, target_idx=0, tau=0.8)
    assert 0 in neighbors
    assert 1 in neighbors
    assert 2 not in neighbors


def test_cluster_by_neighbors_distinct_features():
    """Representations with same dominant feature should cluster together."""
    # Create representations: 3 dominated by f1, 2 dominated by f2
    f1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    f2 = torch.tensor([0.0, 1.0, 0.0, 0.0])

    representations = torch.stack([
        f1 + 0.1 * torch.randn(4),  # 0: dominated by f1
        f1 + 0.1 * torch.randn(4),  # 1: dominated by f1
        f1 + 0.1 * torch.randn(4),  # 2: dominated by f1
        f2 + 0.1 * torch.randn(4),  # 3: dominated by f2
        f2 + 0.1 * torch.randn(4),  # 4: dominated by f2
    ])
    representations = representations / representations.norm(dim=1, keepdim=True)

    clusters = cluster_by_neighbors(representations, tau=0.8)

    # Should have 2 clusters
    assert len(clusters) == 2


def test_cluster_by_neighbors_returns_dict():
    representations = torch.eye(3)
    clusters = cluster_by_neighbors(representations, tau=0.5)

    # Each orthogonal vector forms its own cluster
    assert isinstance(clusters, dict)
    assert len(clusters) == 3
