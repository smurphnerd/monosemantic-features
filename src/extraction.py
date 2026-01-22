import torch
from src.config import SyntheticConfig, ExtractionConfig, compute_coef_min


def compute_tau_bounds(
    k: int,
    epsilon: float,
    coef_min: float,
    coef_max: float
) -> tuple[float, float]:
    """
    Compute valid τ range based on generative parameters.

    Args:
        k: Maximum number of active features per representation
        epsilon: Orthogonality tolerance (|⟨f_i, f_j⟩| ≤ ε for i≠j)
        coef_min: Minimum coefficient magnitude
        coef_max: Maximum coefficient magnitude

    Returns:
        tau_upper: Upper bound on τ (min similarity when sharing a feature)
        tau_lower: Lower bound on τ (max similarity when NOT sharing)

    For τ to separate sharing from non-sharing, we need tau_lower < τ < tau_upper.
    """
    # Max norm per representation: sqrt(k * coef_max^2)
    max_norm_sq = k * coef_max ** 2

    # tau_upper: minimum similarity when SHARING a feature
    # Worst case: minimum coefficients on shared feature, maximum norms
    # cossim = (c1 * c2) / (||r1|| * ||r2||) ≥ coef_min^2 / max_norm^2
    tau_upper = coef_min * coef_min / max_norm_sq

    # tau_lower: maximum similarity when NOT sharing any feature
    # Only ε-interference between k features in each representation
    # Max dot product: k * k * coef_max^2 * ε (all pairs ε-correlated)
    # For ε=0: tau_lower = 0 (orthogonal features have zero interference)
    tau_lower = k * k * coef_max * coef_max * epsilon / max_norm_sq

    return tau_upper, tau_lower


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
    tau_upper, tau_lower = compute_tau_bounds(
        k=synthetic_config.k,
        epsilon=synthetic_config.epsilon,
        coef_min=coef_min,
        coef_max=synthetic_config.coef_max
    )

    # Interpolate between bounds using tau_margin
    # margin=0 → tau_lower, margin=1 → tau_upper, margin=0.5 → midpoint
    margin = extraction_config.tau_margin
    tau = tau_lower + margin * (tau_upper - tau_lower)

    return tau


def find_neighbors(
    representations: torch.Tensor,
    target_idx: int,
    tau: float,
    norms: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Find indices of representations with |cosine similarity| >= tau to target.

    Uses absolute cosine similarity to catch both aligned and anti-aligned
    representations that share a feature (regardless of coefficient sign).

    Args:
        representations: (num_repr, d) tensor - rows are representation vectors
        target_idx: Index of target representation
        tau: Cosine similarity threshold (applied to absolute value)
        norms: Precomputed norms (num_repr,) tensor, or None to compute

    Returns:
        1D tensor of neighbor indices (always includes target_idx)
    """
    target = representations[target_idx]

    # Use precomputed norms if provided
    if norms is None:
        norms = torch.norm(representations, dim=1)
    target_norm = norms[target_idx]

    dots = representations @ target
    cosine_sims = dots / (norms * target_norm + 1e-8)

    # Find neighbors using absolute cosine similarity
    # This catches both aligned (+) and anti-aligned (-) representations
    neighbor_mask = torch.abs(cosine_sims) >= tau
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
        representations: (num_repr, d) tensor - rows are representation vectors
        tau: Cosine similarity threshold for neighbors

    Returns:
        Dictionary mapping neighbor_set (frozenset) to list of representation indices
    """
    num_repr = representations.shape[0]
    clusters: dict[frozenset[int], list[int]] = {}

    # Precompute norms once for all representations
    norms = torch.norm(representations, dim=1)

    for i in range(num_repr):
        neighbor_indices = find_neighbors(representations, i, tau, norms=norms)
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
       a. Compute ε-nullspace of mean-centered non-neighbors
       b. Project ALL neighbor representations onto nullspace
       c. SVD to get dominant direction → extracted feature
    3. Deduplicate similar features
    4. Return all extracted features

    Args:
        representations: (num_repr, d) tensor - rows are representation vectors
        extraction_config: ExtractionConfig with tau and epsilon
        synthetic_config: SyntheticConfig for resolving tau

    Returns:
        (m, d) tensor of extracted features - rows are unit-norm feature vectors
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
