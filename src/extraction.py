import torch
from src.config import SyntheticConfig, ExtractionConfig, compute_coef_min


def compute_tau_bounds(
    k: int,
    epsilon: float,
    coef_min: float,
    coef_max: float
) -> tuple[float, float]:
    """
    Compute valid τ bounds based on generative parameters.

    For two representations sharing feature f_i with coefficients c1, c2:
    - Their dot product includes c1*c2 (shared feature contribution)
    - Plus interference from other features (ε-correlated)

    Args:
        k: Maximum number of active features per representation
        epsilon: Orthogonality tolerance
        coef_min: Minimum coefficient magnitude
        coef_max: Maximum coefficient magnitude

    Returns:
        tau_min: Worst-case cosine similarity when sharing a feature
        tau_max: Best-case cosine similarity when NOT sharing a feature
    """
    # Worst case for sharing: minimum coefficients on shared feature,
    # maximum interference from k-1 other features
    # For orthogonal (epsilon=0): interference is 0
    # Shared contribution: coef_min * coef_min
    # Max norm per representation: sqrt(k * coef_max^2)

    min_shared_dot = coef_min * coef_min
    max_interference = (k - 1) * (k - 1) * coef_max * coef_max * epsilon
    max_norm = (k * coef_max ** 2) ** 0.5

    # tau_min: minimum similarity when sharing
    # Worst case: min shared contribution, max norms
    tau_min = min_shared_dot / (max_norm * max_norm)

    # tau_max: maximum similarity when NOT sharing
    # Best case for non-sharing: all k features ε-correlated
    # Each pair contributes up to coef_max^2 * epsilon
    tau_max = k * k * coef_max * coef_max * epsilon / (max_norm * max_norm)

    # For epsilon=0, tau_max should be 0
    return tau_min, tau_max


def resolve_tau(
    extraction_config: ExtractionConfig,
    synthetic_config: SyntheticConfig
) -> float:
    """
    Resolve the τ threshold: auto-derive if None, otherwise use manual value.

    Args:
        extraction_config: ExtractionConfig with tau and tau_margin
        synthetic_config: SyntheticConfig for deriving bounds

    Returns:
        Resolved τ value
    """
    if extraction_config.tau is not None:
        return extraction_config.tau

    coef_min = compute_coef_min(synthetic_config)
    tau_min, tau_max = compute_tau_bounds(
        k=synthetic_config.k,
        epsilon=synthetic_config.epsilon,
        coef_min=coef_min,
        coef_max=synthetic_config.coef_max
    )

    # Interpolate between bounds using tau_margin
    # margin=0 → tau_max, margin=1 → tau_min, margin=0.5 → midpoint
    margin = extraction_config.tau_margin
    tau = tau_max + margin * (tau_min - tau_max)

    return tau


def find_neighbors(
    representations: torch.Tensor,
    target_idx: int,
    tau: float
) -> torch.Tensor:
    """
    Find indices of representations with cosine similarity >= tau to target.

    Args:
        representations: (num_repr, d) tensor
        target_idx: Index of target representation
        tau: Cosine similarity threshold

    Returns:
        1D tensor of neighbor indices (always includes target_idx)
    """
    target = representations[target_idx]

    # Compute cosine similarities
    norms = torch.norm(representations, dim=1)
    target_norm = norms[target_idx]

    dots = representations @ target
    cosine_sims = dots / (norms * target_norm + 1e-8)

    # Find neighbors (cosine sim >= tau)
    neighbor_mask = cosine_sims >= tau
    neighbor_indices = torch.where(neighbor_mask)[0]

    return neighbor_indices


def cluster_by_neighbors(
    representations: torch.Tensor,
    tau: float
) -> dict[frozenset[int], list[int]]:
    """
    Group representation indices by their neighbor sets.

    Representations with identical neighbor sets are assumed to share
    the same dominant feature.

    Args:
        representations: (num_repr, d) tensor
        tau: Cosine similarity threshold for neighbors

    Returns:
        Dictionary mapping neighbor_set (frozenset) to list of representation indices
    """
    num_repr = representations.shape[0]
    clusters: dict[frozenset[int], list[int]] = {}

    for i in range(num_repr):
        neighbor_indices = find_neighbors(representations, i, tau)
        neighbor_set = frozenset(neighbor_indices.tolist())

        if neighbor_set not in clusters:
            clusters[neighbor_set] = []
        clusters[neighbor_set].append(i)

    return clusters


def compute_nullspace(
    representations: torch.Tensor,
    neighbor_indices: torch.Tensor,
    epsilon: float
) -> torch.Tensor:
    """
    Compute ε-nullspace of non-neighbor representations.

    The nullspace contains directions orthogonal to non-neighbors,
    which should include the feature shared by neighbors.

    Args:
        representations: (num_repr, d) tensor
        neighbor_indices: 1D tensor of neighbor indices
        epsilon: Noise threshold (0 for exact nullspace)

    Returns:
        (k, d) tensor of nullspace basis vectors (unit norm, orthogonal)
    """
    num_repr, d = representations.shape

    # Get non-neighbor indices
    all_indices = set(range(num_repr))
    neighbor_set = set(neighbor_indices.tolist())
    non_neighbor_indices = list(all_indices - neighbor_set)

    if len(non_neighbor_indices) == 0:
        # No non-neighbors: nullspace is entire space
        return torch.eye(d)

    # Stack non-neighbor representations
    non_neighbors = representations[non_neighbor_indices]  # (m, d)

    # SVD to find nullspace
    # non_neighbors = U @ S @ Vh
    # Nullspace is spanned by rows of Vh with singular values < epsilon
    U, S, Vh = torch.linalg.svd(non_neighbors, full_matrices=True)

    # For epsilon=0, nullspace is where S=0 (or very small)
    # Vh has shape (d, d), rows are right singular vectors
    threshold = epsilon if epsilon > 0 else 1e-6

    # Find dimensions where singular values are below threshold
    # S has length min(m, d), pad with zeros if needed
    full_S = torch.zeros(d)
    full_S[:len(S)] = S

    nullspace_mask = full_S < threshold
    nullspace_basis = Vh[nullspace_mask]  # (k, d)

    return nullspace_basis
