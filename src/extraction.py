import torch
from src.config import SyntheticConfig, ExtractionConfig, compute_coef_min
from src.logging_utils import (
    logger,
    log_nullspace_computation,
    log_feature_extraction,
    log_cluster_summary,
    log_deduplication,
)


def build_neighbor_matrix(representations: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Build boolean adjacency matrix for neighbor relationships.

    Args:
        representations: (num_repr, d) tensor - rows are representation vectors
        tau: Cosine similarity threshold (applied to absolute value)

    Returns:
        (num_repr, num_repr) boolean tensor where matrix[i,j] = |cossim(i,j)| >= tau
        Diagonal is True (self-neighbors). Matrix is symmetric.
    """
    # Normalize rows
    norms = torch.norm(representations, dim=1, keepdim=True)
    X_norm = representations / (norms + 1e-8)

    # Pairwise cosine similarities
    cossim = X_norm @ X_norm.T

    # Threshold with absolute value
    neighbor_matrix = torch.abs(cossim) >= tau

    return neighbor_matrix


def find_monosemantic_targets(neighbor_matrix: torch.Tensor) -> torch.Tensor:
    """
    Find representations whose neighbor set size is a local minimum.

    These are likely monosemantic (containing a single feature) and are
    good candidates for clean feature extraction.

    Criterion: |X_pos| <= |X_pos,i| for all neighbors r_i in X_pos

    Args:
        neighbor_matrix: (n, n) boolean tensor from build_neighbor_matrix

    Returns:
        1D tensor of indices for representations passing the minimality criterion.
        Deduplicated: only one representative per unique neighbor set.
    """
    n = neighbor_matrix.shape[0]
    neighbor_counts = neighbor_matrix.sum(dim=1)

    # Group by identical neighbor sets (rows)
    seen_rows: dict[bytes, int] = {}
    representatives: list[int] = []

    for i in range(n):
        row_key = neighbor_matrix[i].cpu().numpy().tobytes()
        if row_key not in seen_rows:
            seen_rows[row_key] = i
            representatives.append(i)

    # Filter by minimality criterion
    targets: list[int] = []
    for i in representatives:
        neighbors = neighbor_matrix[i]
        if not neighbors.any():
            # No neighbors - vacuously satisfies minimality criterion
            targets.append(i)
            continue
        neighbor_count_values = neighbor_counts[neighbors]
        min_neighbor_count = neighbor_count_values.min()
        if neighbor_counts[i] <= min_neighbor_count:
            targets.append(i)

    return torch.tensor(targets, dtype=torch.long)


def compute_tau_bounds(
    k: int, epsilon: float, coef_min: float, coef_max: float
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
    max_norm_sq = k * coef_max**2

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
    extraction_config: ExtractionConfig, synthetic_config: SyntheticConfig, epsilon: float = 0.0
) -> float:
    """
    Resolve the τ threshold: auto-derive if None, otherwise use manual value.

    Args:
        extraction_config: ExtractionConfig with tau and tau_margin
        synthetic_config: SyntheticConfig for deriving bounds
        epsilon: Orthogonality tolerance from feature basis generation

    Returns:
        Resolved τ value
    """
    if extraction_config.tau is not None:
        return extraction_config.tau

    coef_min = compute_coef_min(synthetic_config, epsilon)
    tau_upper, tau_lower = compute_tau_bounds(
        k=synthetic_config.k,
        epsilon=epsilon,
        coef_min=coef_min,
        coef_max=synthetic_config.coef_max,
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
    norms: torch.Tensor | None = None,
    max_neighbors: int | None = None,
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
        max_neighbors: If set, limit to top n neighbors by cosine similarity

    Returns:
        1D tensor of neighbor indices (always includes target_idx)
    """
    target = representations[target_idx]

    # Use precomputed norms if provided
    if norms is None:
        norms = torch.norm(representations, dim=1)
    assert norms is not None
    target_norm = norms[target_idx]

    dots = representations @ target
    cosine_sims = dots / (norms * target_norm + 1e-8)

    # Find neighbors using absolute cosine similarity
    # This catches both aligned (+) and anti-aligned (-) representations
    neighbor_mask = torch.abs(cosine_sims) >= tau
    neighbor_indices = torch.where(neighbor_mask)[0]

    # Limit to top n neighbors by cosine similarity if max_neighbors is set
    if max_neighbors is not None and len(neighbor_indices) > max_neighbors:
        neighbor_sims = cosine_sims[neighbor_indices]
        _, top_indices = torch.topk(neighbor_sims, max_neighbors)
        neighbor_indices = neighbor_indices[top_indices]

    return neighbor_indices


def cluster_by_neighbors(
    representations: torch.Tensor,
    tau: float,
    max_neighbors: int | None = None,
) -> dict[frozenset[int], list[int]]:
    """
    Group representation indices by their neighbor sets.

    Representations with identical neighbor sets are assumed to share
    the same dominant feature.

    Args:
        representations: (num_repr, d) tensor - rows are representation vectors
        tau: Cosine similarity threshold for neighbors
        max_neighbors: If set, limit neighbor sets to top n by cosine similarity

    Returns:
        Dictionary mapping neighbor_set (frozenset) to list of representation indices
    """
    num_repr = representations.shape[0]
    clusters: dict[frozenset[int], list[int]] = {}

    # Precompute norms once for all representations
    norms = torch.norm(representations, dim=1)

    for i in range(num_repr):
        neighbor_indices = find_neighbors(
            representations, i, tau, norms=norms, max_neighbors=max_neighbors
        )
        neighbor_set = frozenset(neighbor_indices.tolist())

        if neighbor_set not in clusters:
            clusters[neighbor_set] = []
        clusters[neighbor_set].append(i)

    return clusters


def compute_nullspace(
    representations: torch.Tensor,
    neighbor_indices: torch.Tensor,
    epsilon: float,
    max_non_neighbors: int | None = None,
    neg_tau: float | None = None,
    verbose: bool = False,
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
        max_non_neighbors: If set, limit to n non-neighbors with lowest |cosine sim|
        neg_tau: If set, only include non-neighbors with |cosine sim| < neg_tau
        verbose: If True, log computation details

    Returns:
        (k, d) tensor of nullspace basis vectors - rows are unit-norm orthogonal vectors
    """
    num_repr, d = representations.shape
    neighbor_set = set(neighbor_indices.tolist())

    # Compute cosine similarity to neighbor mean for filtering
    neighbors = representations[neighbor_indices]
    neighbor_mean = neighbors.mean(dim=0)
    neighbor_mean_norm = torch.norm(neighbor_mean)

    all_norms = torch.norm(representations, dim=1)
    all_dots = representations @ neighbor_mean
    all_cosine_sims = torch.abs(all_dots / (all_norms * neighbor_mean_norm + 1e-8))

    # Get non-neighbor indices, optionally filtered by neg_tau
    non_neighbor_indices = []
    for i in range(num_repr):
        if i in neighbor_set:
            continue
        if neg_tau is not None and all_cosine_sims[i] > neg_tau:
            continue  # Skip if similarity too high
        non_neighbor_indices.append(i)

    if len(non_neighbor_indices) == 0:
        # No valid non-neighbors: nullspace is entire space
        return torch.eye(d)

    # Further limit to lowest similarity if max_non_neighbors is set
    if max_non_neighbors is not None and len(non_neighbor_indices) > max_non_neighbors:
        non_neighbor_tensor = torch.tensor(non_neighbor_indices)
        cosine_sims = all_cosine_sims[non_neighbor_tensor]
        _, bottom_indices = torch.topk(cosine_sims, max_non_neighbors, largest=False)
        non_neighbor_indices = non_neighbor_tensor[bottom_indices].tolist()

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
    # RMS of representation norms (use all representations for stability)
    norms = torch.norm(representations, dim=1)
    rms_norm = torch.sqrt(torch.mean(norms**2))
    if epsilon > 0:
        scaled_epsilon = (n_neg**0.5) * rms_norm * epsilon
    else:
        # For epsilon=0, use small threshold for numerical stability
        scaled_epsilon = 1e-6

    # Find rows of Vh where singular values are below scaled threshold
    # S has length min(n_neg, d), pad with zeros for remaining directions
    full_S = torch.zeros(d)
    full_S[: len(S)] = S

    nullspace_mask = full_S < scaled_epsilon
    nullspace_basis = Vh[nullspace_mask]  # (k, d) - rows are basis vectors

    if verbose:
        log_nullspace_computation(n_neg, float(rms_norm), epsilon, float(scaled_epsilon), full_S, int(nullspace_mask.sum()))

    return nullspace_basis


def extract_feature(
    representations: torch.Tensor,
    neighbor_indices: torch.Tensor,
    nullspace: torch.Tensor,
    verbose: bool = False,
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
        verbose: If True, log extraction details

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
    _, S, Vh = torch.linalg.svd(projected, full_matrices=False)

    if verbose:
        log_feature_extraction(len(neighbor_indices), S)

    # First right singular vector is the feature (already unit norm)
    feature = Vh[0]  # (d,)

    return feature


def extract_all_features(
    representations: torch.Tensor,
    extraction_config: ExtractionConfig,
    synthetic_config: SyntheticConfig,
    verbose: bool = False,
    use_minimality_filter: bool = True,
    basis_epsilon: float = 0.0,
) -> torch.Tensor:
    """
    Extract all monosemantic features from representations.

    Algorithm:
    1. Build neighbor matrix upfront
    2. Select target representations:
       - If use_minimality_filter: select monosemantic targets (local minima in neighbor count)
       - Otherwise: use all unique neighbor set representatives
    3. For each target:
       a. Compute ε-nullspace of mean-centered non-neighbors
       b. Project ALL neighbor representations onto nullspace
       c. SVD to get dominant direction → extracted feature
    4. Deduplicate similar features
    5. Return all extracted features

    Args:
        representations: (num_repr, d) tensor - rows are representation vectors
        extraction_config: ExtractionConfig with tau and epsilon
        synthetic_config: SyntheticConfig for resolving tau
        verbose: If True, log extraction details
        use_minimality_filter: If True, only extract from monosemantic targets
            (representations whose neighbor count is a local minimum)
        basis_epsilon: Orthogonality tolerance from feature basis generation
            (used for auto-deriving tau threshold)

    Returns:
        (m, d) tensor of extracted features - rows are unit-norm feature vectors
    """
    tau = resolve_tau(extraction_config, synthetic_config, basis_epsilon)
    epsilon = extraction_config.epsilon
    max_neighbors = extraction_config.max_neighbors
    neg_tau = extraction_config.neg_tau

    # Step 1: Build neighbor matrix upfront
    neighbor_matrix = build_neighbor_matrix(representations, tau)

    # Step 2: Select target representations
    if use_minimality_filter:
        # New path: only monosemantic targets (local minima in neighbor count)
        target_indices = find_monosemantic_targets(neighbor_matrix)
    else:
        # Old path: all unique neighbor set representatives
        clusters = cluster_by_neighbors(representations, tau, max_neighbors=max_neighbors)
        # Get one representative per cluster
        target_indices = torch.tensor([indices[0] for indices in clusters.values()])

    if verbose:
        log_cluster_summary(len(target_indices), [int(neighbor_matrix[i].sum()) for i in target_indices])

    # Step 3: Extract feature from each target
    extracted_features = []

    for idx in target_indices:
        idx = int(idx)
        neighbor_mask = neighbor_matrix[idx]
        neighbor_indices = torch.where(neighbor_mask)[0]

        if len(neighbor_indices) < 2:
            # Skip targets with only one neighbor (likely noise)
            continue

        try:
            nullspace = compute_nullspace(
                representations,
                neighbor_indices,
                epsilon,
                max_non_neighbors=max_neighbors,
                neg_tau=neg_tau,
                verbose=verbose,
            )
            if nullspace.shape[0] == 0:
                continue
            feature = extract_feature(representations, neighbor_indices, nullspace, verbose=verbose)
            extracted_features.append(feature)
        except ValueError:
            # Skip if extraction fails (e.g., zero projection)
            continue

    if len(extracted_features) == 0:
        return torch.empty(0, representations.shape[1])

    features = torch.stack(extracted_features)  # (m, d)

    # Deduplicate similar features (different targets can extract same feature)
    deduplicated = _deduplicate_features(features, threshold=0.99)

    if verbose:
        log_deduplication(features.shape[0], deduplicated.shape[0])

    return deduplicated


def _deduplicate_features(
    features: torch.Tensor, threshold: float = 0.99
) -> torch.Tensor:
    """Remove near-duplicate features based on cosine similarity."""
    if features.shape[0] <= 1:
        return features

    # features are unit norm, so dot product = cosine sim
    sims = torch.abs(features @ features.T)

    # Greedy deduplication: keep feature if not too similar to any kept feature
    keep_mask = torch.ones(features.shape[0], dtype=torch.bool)

    for i in range(features.shape[0]):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, features.shape[0]):
            if sims[i, j] > threshold:
                keep_mask[j] = False

    return features[keep_mask]
