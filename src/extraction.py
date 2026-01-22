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
    Compute ε-nullspace of mean-centered non-neighbor representations.

    The nullspace contains directions orthogonal to non-neighbors,
    which should include the feature shared by neighbors.

    Following the methodology:
    1. Mean-center non-neighbors: X̄_neg = X_neg - μ_neg
    2. Compute scaled epsilon: ε̃ = √(n_neg) · RMS(‖r‖₂) · ε
    3. SVD to find directions with σ_i < ε̃

    Convention: All matrices use row-major format (rows are vectors).

    Args:
        representations: (num_repr, d) tensor - rows are representation vectors
        neighbor_indices: 1D tensor of neighbor indices
        epsilon: Noise threshold (0 for exact nullspace)

    Returns:
        (k, d) tensor of nullspace basis vectors - rows are unit-norm orthogonal vectors
    """
    num_repr, d = representations.shape

    # Get non-neighbor indices
    all_indices = set(range(num_repr))
    neighbor_set = set(neighbor_indices.tolist())
    non_neighbor_indices = list(all_indices - neighbor_set)

    if len(non_neighbor_indices) == 0:
        # No non-neighbors: nullspace is entire space
        return torch.eye(d)

    # Stack non-neighbor representations: (n_neg, d) - rows are vectors
    non_neighbors = representations[non_neighbor_indices]
    n_neg = len(non_neighbor_indices)

    # Mean-center the non-neighbors
    mean_neg = non_neighbors.mean(dim=0, keepdim=True)  # (1, d)
    non_neighbors_centered = non_neighbors - mean_neg  # (n_neg, d)

    # SVD on row-major data: X = U @ diag(S) @ Vh
    # Vh rows are principal directions in d-space
    # Small singular values → nullspace directions
    U, S, Vh = torch.linalg.svd(non_neighbors_centered, full_matrices=True)
    # U: (n_neg, n_neg), S: (min(n_neg, d),), Vh: (d, d)

    # Compute scaled epsilon threshold:
    # ε̃ = √(n_neg) · RMS(‖r‖₂) · ε
    if epsilon > 0:
        # RMS of representation norms (use all representations for stability)
        norms = torch.norm(representations, dim=1)
        rms_norm = torch.sqrt(torch.mean(norms ** 2))
        scaled_epsilon = (n_neg ** 0.5) * rms_norm * epsilon
    else:
        # For epsilon=0, use small threshold for numerical stability
        scaled_epsilon = 1e-6

    # Find rows of Vh where singular values are below scaled threshold
    # S has length min(n_neg, d), pad with zeros for remaining directions
    full_S = torch.zeros(d)
    full_S[:len(S)] = S

    nullspace_mask = full_S < scaled_epsilon
    nullspace_basis = Vh[nullspace_mask]  # (k, d) - rows are basis vectors

    return nullspace_basis


def extract_feature(
    representations: torch.Tensor,
    neighbor_indices: torch.Tensor,
    nullspace: torch.Tensor
) -> torch.Tensor:
    """
    Extract feature by projecting neighbors onto nullspace and finding dominant direction.

    Following the methodology:
    f = SVD₁(U_low U_low^T X_pos)

    Projects ALL neighbor representations onto the nullspace, then uses SVD
    to find the direction of maximum variance.

    Convention: All matrices use row-major format (rows are vectors).

    Args:
        representations: (num_repr, d) tensor - rows are representation vectors
        neighbor_indices: 1D tensor of neighbor indices
        nullspace: (k, d) tensor of nullspace basis vectors - rows are basis vectors

    Returns:
        (d,) unit-norm feature vector
    """
    if nullspace.shape[0] == 0:
        raise ValueError("Nullspace is empty, cannot extract feature")

    # Get neighbor representations: (m, d) - rows are vectors
    neighbors = representations[neighbor_indices]

    # Project ALL neighbors onto nullspace
    # For row vectors: projected = X @ nullspace.T @ nullspace
    # This applies the projection matrix P = nullspace.T @ nullspace to each row
    projected = neighbors @ nullspace.T @ nullspace  # (m, d)

    # SVD to find direction of maximum variance
    # projected = U @ diag(S) @ Vh
    # Vh[0] (first row) is the dominant direction in d-space
    U, S, Vh = torch.linalg.svd(projected, full_matrices=False)

    # First right singular vector is the feature (already unit norm)
    feature = Vh[0]  # (d,)

    return feature


def extract_all_features(
    representations: torch.Tensor,
    extraction_config: ExtractionConfig,
    synthetic_config: SyntheticConfig
) -> torch.Tensor:
    """
    Extract all monosemantic features from representations.

    Algorithm:
    1. Cluster representations by identical neighbor sets
    2. For each cluster:
       a. Average the representations in the cluster
       b. Compute nullspace of non-neighbors
       c. Project averaged representation onto nullspace
       d. SVD to get dominant direction → extracted feature
    3. Deduplicate similar features
    4. Return all extracted features

    Args:
        representations: (num_repr, d) tensor
        extraction_config: ExtractionConfig with tau and epsilon
        synthetic_config: SyntheticConfig for resolving tau

    Returns:
        (m, d) tensor of extracted features
    """
    tau = resolve_tau(extraction_config, synthetic_config)
    epsilon = extraction_config.epsilon

    # Step 1: Cluster by neighbor sets
    clusters = cluster_by_neighbors(representations, tau)

    # Step 2: Extract feature from each cluster
    extracted_features = []

    for neighbor_set, indices in clusters.items():
        if len(neighbor_set) < 2:
            # Skip clusters with only one member (likely noise)
            continue

        neighbor_indices = torch.tensor(list(neighbor_set))

        try:
            nullspace = compute_nullspace(representations, neighbor_indices, epsilon)
            if nullspace.shape[0] == 0:
                continue
            feature = extract_feature(representations, neighbor_indices, nullspace)
            extracted_features.append(feature)
        except ValueError:
            # Skip if extraction fails (e.g., zero projection)
            continue

    if len(extracted_features) == 0:
        return torch.empty(0, representations.shape[1])

    # Stack features
    features = torch.stack(extracted_features)  # (m, d)

    # Step 3: Deduplicate similar features (cosine similarity > 0.99)
    deduplicated = _deduplicate_features(features, threshold=0.99)

    return deduplicated


def _deduplicate_features(
    features: torch.Tensor,
    threshold: float = 0.99
) -> torch.Tensor:
    """Remove near-duplicate features based on cosine similarity."""
    if features.shape[0] <= 1:
        return features

    # Compute pairwise cosine similarities
    # features are unit norm, so dot product = cosine sim
    sims = torch.abs(features @ features.T)  # Absolute for sign invariance

    # Greedy deduplication: keep feature if not too similar to any kept feature
    keep_mask = torch.ones(features.shape[0], dtype=torch.bool)

    for i in range(features.shape[0]):
        if not keep_mask[i]:
            continue
        # Mark all subsequent similar features as duplicates
        for j in range(i + 1, features.shape[0]):
            if sims[i, j] > threshold:
                keep_mask[j] = False

    return features[keep_mask]
