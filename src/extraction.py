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
