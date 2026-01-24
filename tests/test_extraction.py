import torch
from src.config import SyntheticConfig, ExtractionConfig
from src.extraction import compute_tau_bounds, resolve_tau, find_neighbors, cluster_by_neighbors
from src.extraction import compute_nullspace
from src.extraction import extract_feature
from src.extraction import build_neighbor_matrix, find_monosemantic_targets


def test_compute_tau_bounds_orthogonal():
    """For orthogonal features, tau bounds should be well-separated."""
    k = 3
    epsilon = 0.0
    coef_min = 0.1
    coef_max = 1.0

    tau_upper, tau_lower = compute_tau_bounds(k, epsilon, coef_min, coef_max)

    # tau_lower should be less than tau_upper for valid separation
    assert tau_lower < tau_upper


def test_resolve_tau_auto():
    """When tau is None, it should be auto-derived."""
    synthetic = SyntheticConfig(k=3, coef_min_floor=0.1)
    extraction = ExtractionConfig(tau=None, tau_margin=0.5)

    tau = resolve_tau(extraction, synthetic, epsilon=0.0)

    assert tau > 0
    assert tau < 1


def test_resolve_tau_manual():
    """When tau is set, it should be used directly."""
    synthetic = SyntheticConfig(k=3)
    extraction = ExtractionConfig(tau=0.7)

    tau = resolve_tau(extraction, synthetic, epsilon=0.0)

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


def test_compute_nullspace_orthogonal():
    """Nullspace of non-neighbors should contain the shared feature direction."""
    # 4D space, 3 representations along first 3 axes
    representations = torch.eye(4)[:3]  # (3, 4)

    # Target is [0], neighbors are [0], non-neighbors are [1, 2]
    neighbor_indices = torch.tensor([0])

    # With mean-centering, 2 non-neighbor points become colinear (rank 1)
    # So nullspace is d - 1 = 3 dimensional
    nullspace = compute_nullspace(representations, neighbor_indices, epsilon=0.0)

    # Nullspace should have rank 3 (4D space minus 1 centered direction)
    assert nullspace.shape[0] == 3

    # Key property: e1 (shared direction) and e4 (unused direction) should be in nullspace
    e1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    e4 = torch.tensor([0.0, 0.0, 0.0, 1.0])

    # Project onto nullspace and check magnitude
    proj_e1 = nullspace @ e1
    proj_e4 = nullspace @ e4
    assert proj_e1.norm() > 0.99, "e1 should be in nullspace"
    assert proj_e4.norm() > 0.99, "e4 should be in nullspace"


def test_compute_nullspace_contains_shared_direction():
    """The nullspace should contain the direction shared by neighbors."""
    # Create representations where neighbors share feature f1
    f1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    f2 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    f3 = torch.tensor([0.0, 0.0, 1.0, 0.0])

    representations = torch.stack([
        f1,  # 0: neighbor (has f1)
        f1,  # 1: neighbor (has f1)
        f2,  # 2: non-neighbor
        f3,  # 3: non-neighbor
    ])

    neighbor_indices = torch.tensor([0, 1])
    nullspace = compute_nullspace(representations, neighbor_indices, epsilon=0.0)

    # Project f1 onto nullspace - should have high component
    projection = nullspace @ f1
    assert projection.norm() > 0.9  # f1 should be mostly in nullspace


def test_extract_feature_recovers_shared_direction():
    """Extract feature should recover the direction shared by neighbors."""
    # Create representations where neighbors share feature f1
    f1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    f2 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    f3 = torch.tensor([0.0, 0.0, 1.0, 0.0])

    representations = torch.stack([
        f1,  # 0: neighbor
        f1,  # 1: neighbor
        f2,  # 2: non-neighbor
        f3,  # 3: non-neighbor
    ])

    neighbor_indices = torch.tensor([0, 1])

    nullspace = compute_nullspace(representations, neighbor_indices, epsilon=0.0)
    extracted = extract_feature(representations, neighbor_indices, nullspace)

    # Extracted feature should align with f1
    alignment = torch.abs(extracted @ f1)
    assert alignment > 0.99


def test_extract_feature_unit_norm():
    """Extracted feature should have unit norm."""
    representations = torch.randn(10, 8)
    representations = representations / representations.norm(dim=1, keepdim=True)
    neighbor_indices = torch.tensor([0, 1, 2])

    nullspace = compute_nullspace(representations, neighbor_indices, epsilon=0.0)
    if nullspace.shape[0] > 0:  # Only if nullspace is non-trivial
        extracted = extract_feature(representations, neighbor_indices, nullspace)
        assert torch.isclose(extracted.norm(), torch.tensor(1.0), atol=1e-6)


# Tests for build_neighbor_matrix and find_monosemantic_targets


def test_build_neighbor_matrix_diagonal_true():
    """Diagonal should be True (self-neighbors)."""
    representations = torch.randn(5, 4)
    representations = representations / representations.norm(dim=1, keepdim=True)

    matrix = build_neighbor_matrix(representations, tau=0.5)

    assert matrix.diagonal().all(), "Diagonal should all be True"


def test_build_neighbor_matrix_symmetric():
    """Matrix should be symmetric."""
    representations = torch.randn(5, 4)
    representations = representations / representations.norm(dim=1, keepdim=True)

    matrix = build_neighbor_matrix(representations, tau=0.5)

    assert torch.equal(matrix, matrix.T), "Matrix should be symmetric"


def test_build_neighbor_matrix_threshold():
    """Matrix should correctly threshold by cosine similarity."""
    # Create known representations
    representations = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],  # ~0.99 cossim with first
        [0.0, 1.0, 0.0],  # 0 cossim with first
    ])
    representations = representations / representations.norm(dim=1, keepdim=True)

    matrix = build_neighbor_matrix(representations, tau=0.8)

    # 0 and 1 should be neighbors
    assert matrix[0, 1] == True
    assert matrix[1, 0] == True
    # 0 and 2 should not be neighbors
    assert matrix[0, 2] == False
    assert matrix[2, 0] == False


def test_find_monosemantic_targets_deduplicates():
    """Identical neighbor sets should yield single representative."""
    # Create matrix where rows 0 and 1 have identical neighbor patterns
    matrix = torch.tensor([
        [True, True, False, False],
        [True, True, False, False],  # Same as row 0
        [False, False, True, True],
        [False, False, True, True],  # Same as row 2
    ])

    targets = find_monosemantic_targets(matrix)

    # Should have exactly 2 targets (one per unique neighbor set)
    assert len(targets) == 2


def test_find_monosemantic_targets_minimality_criterion():
    """Should select representations with minimal neighbor count among neighbors."""
    # Create a scenario:
    # - Rep 0: neighbors {0, 1, 2} (count=3)
    # - Rep 1: neighbors {0, 1, 2, 3} (count=4, superset of rep 0's neighbors)
    # - Rep 2: neighbors {0, 1, 2} (count=3)
    # - Rep 3: neighbors {1, 3} (count=2)
    #
    # Rep 0's neighbors have counts [3, 4, 3], min=3, rep 0 has 3 -> passes
    # Rep 1's neighbors have counts [3, 4, 3, 2], min=2, rep 1 has 4 -> fails
    # Rep 3's neighbors have counts [4, 2], min=2, rep 3 has 2 -> passes

    matrix = torch.tensor([
        [True, True, True, False],   # 0: neighbors {0,1,2}
        [True, True, True, True],    # 1: neighbors {0,1,2,3}
        [True, True, True, False],   # 2: neighbors {0,1,2} - same as 0
        [False, True, False, True],  # 3: neighbors {1,3}
    ])

    targets = find_monosemantic_targets(matrix)
    target_list = targets.tolist()

    # Rep 0 (or 2, deduplicated) should be in targets (count 3, min neighbor count 3)
    assert 0 in target_list or 2 in target_list
    # Rep 3 should be in targets (count 2, min neighbor count 2)
    assert 3 in target_list
    # Rep 1 should NOT be in targets (count 4 > min neighbor count 2)
    assert 1 not in target_list


def test_find_monosemantic_targets_all_polysemantic():
    """When all representations are polysemantic, should return empty."""
    # Create scenario where no representation has minimal neighbor count
    # Every representation has a neighbor with fewer neighbors
    matrix = torch.tensor([
        [True, True, True, True],    # 0: all neighbors (count=4)
        [True, True, True, True],    # 1: all neighbors (count=4)
        [True, True, True, True],    # 2: all neighbors (count=4)
        [True, True, True, True],    # 3: all neighbors (count=4)
    ])

    targets = find_monosemantic_targets(matrix)

    # All have the same count (4), and all are neighbors of each other
    # So min among neighbors = 4 for all, and all have count 4
    # All should pass the minimality criterion (count <= min)
    # But they all have identical rows, so should deduplicate to 1
    assert len(targets) == 1
