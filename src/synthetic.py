import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.config import SyntheticConfig, compute_coef_min


@dataclass
class FeatureBasisResult:
    """Result of feature basis generation."""
    features: torch.Tensor      # (n, d) unit-norm feature vectors
    achieved_epsilon: float     # max |<f_i, f_j>| across all pairs
    welch_bound: float          # theoretical minimum for this n, d
    converged: bool             # True if iterative method stabilized


def welch_bound(n: int, d: int) -> float:
    """
    Theoretical minimum achievable max coherence for n unit vectors in d dims.

    Returns:
        0.0 if n <= d (orthonormal basis possible)
        sqrt((n-d) / (d*(n-1))) if n > d
    """
    if n <= d:
        return 0.0
    return math.sqrt((n - d) / (d * (n - 1)))


def _compute_coherence(features: torch.Tensor) -> float:
    """Compute max off-diagonal absolute coherence."""
    G = features @ features.T
    G.fill_diagonal_(0)
    return G.abs().max().item()


def _iterative_projection(
    d: int,
    n: int,
    max_iters: int = 10000,
    patience: int = 500,
) -> FeatureBasisResult:
    """
    Generate n > d features with minimal coherence via vectorized gradient descent.

    Uses soft thresholding: penalizes all correlations above Welch bound,
    with stronger penalty for larger violations.
    """
    # Initialize with random unit vectors
    features = torch.randn(n, d)
    features = F.normalize(features, dim=1)

    target = welch_bound(n, d)
    best_coherence = float('inf')
    best_features = features.clone()
    stall_count = 0
    converged = False

    # Adaptive learning rate based on problem difficulty
    redundancy = n / d
    lr = 0.5 / redundancy  # Smaller lr for higher redundancy

    for _ in range(max_iters):
        # Compute Gram matrix (n x n)
        G = features @ features.T
        G.fill_diagonal_(0)

        curr_coherence = G.abs().max().item()

        # Track best
        if curr_coherence < best_coherence - 1e-6:
            best_coherence = curr_coherence
            best_features = features.clone()
            stall_count = 0
        else:
            stall_count += 1

        # Check convergence
        if stall_count >= patience:
            converged = True
            break

        # Decay learning rate when stuck
        if stall_count > 0 and stall_count % 100 == 0:
            lr *= 0.8

        # Stop if close to Welch bound
        if curr_coherence < target * 1.05:
            converged = True
            break

        # Soft threshold: penalize correlations above target
        # Use squared penalty for smoother gradients
        excess = G.abs() - target
        penalty = torch.clamp(excess, min=0) ** 2
        penalty = penalty * torch.sign(G)

        # Vectorized gradient
        gradient = penalty @ features

        # Normalize gradient to prevent exploding updates
        grad_norm = gradient.norm()
        if grad_norm > 1e-8:
            gradient = gradient / grad_norm

        # Update and normalize
        features = features - lr * gradient
        features = F.normalize(features, dim=1)

    return FeatureBasisResult(
        features=best_features,
        achieved_epsilon=best_coherence,
        welch_bound=target,
        converged=converged
    )


def generate_feature_basis(d: int, n: int) -> FeatureBasisResult:
    """
    Generate n unit-norm feature vectors in d dimensions with minimal coherence.

    Args:
        d: Dimension of representation space
        n: Number of features

    Returns:
        FeatureBasisResult with features and metadata

    For n <= d: Returns orthonormal basis (achieved_epsilon = 0)
    For n > d: Uses Gram matrix alternating projection
    """
    if n <= d:
        # QR decomposition produces orthonormal columns
        random_matrix = torch.randn(d, n)
        q, _ = torch.linalg.qr(random_matrix)
        features = q.T  # (n, d) - rows are unit-norm orthogonal feature vectors

        return FeatureBasisResult(
            features=features,
            achieved_epsilon=0.0,
            welch_bound=0.0,
            converged=True
        )
    else:
        return _iterative_projection(d, n)


def generate_representations(
    features: torch.Tensor,
    config: SyntheticConfig,
    epsilon: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate representations as sparse linear combinations of features.

    Args:
        features: (n, d) tensor of feature vectors
        config: SyntheticConfig with sparsity and coefficient settings
        epsilon: Feature orthogonality tolerance (for coefficient scaling)

    Returns:
        representations: (num_representations, d)
        coefficients: (num_representations, n) ground-truth sparse coefficients
    """
    n, _ = features.shape
    num_repr = config.num_representations
    coef_min = compute_coef_min(config, epsilon)
    coef_max = config.coef_max

    # Initialize coefficients as zeros
    coefficients = torch.zeros(num_repr, n)

    for i in range(num_repr):
        # Determine which features are active based on sparsity mode
        if config.sparsity_mode == "fixed":
            active_indices = torch.randperm(n)[: config.k]

            # Generate coefficients: uniform magnitude in [coef_min, coef_max]
            num_active = len(active_indices)
            magnitudes = torch.rand(num_active) * (coef_max - coef_min) + coef_min

            if config.positive_only:
                coefficients[i, active_indices] = magnitudes
            else:
                signs = torch.randint(0, 2, (num_active,)) * 2 - 1
                coefficients[i, active_indices] = magnitudes * signs

        elif config.sparsity_mode == "variable":
            k_min = config.k_min if config.k_min is not None else 1
            num_active = int(torch.randint(k_min, config.k + 1, (1,)).item())
            active_indices = torch.randperm(n)[:num_active]

            # Generate coefficients: uniform magnitude in [coef_min, coef_max]
            magnitudes = torch.rand(num_active) * (coef_max - coef_min) + coef_min

            if config.positive_only:
                coefficients[i, active_indices] = magnitudes
            else:
                signs = torch.randint(0, 2, (num_active,)) * 2 - 1
                coefficients[i, active_indices] = magnitudes * signs

        elif config.sparsity_mode == "probabilistic":
            p = config.k / n
            active_mask = torch.rand(n) < p
            active_indices = torch.where(active_mask)[0]
            # Ensure at least one feature is active
            if len(active_indices) == 0:
                active_indices = torch.randint(0, n, (1,))

            # Generate coefficients for active features
            num_active = len(active_indices)
            # Uniform magnitude in [coef_min, coef_max]
            magnitudes = torch.rand(num_active) * (coef_max - coef_min) + coef_min

            if config.positive_only:
                coefficients[i, active_indices] = magnitudes
            else:
                # Random signs
                signs = torch.randint(0, 2, (num_active,)) * 2 - 1
                coefficients[i, active_indices] = magnitudes * signs

        elif config.sparsity_mode == "bernoulli_gaussian":
            # Bernoulli-Gaussian distribution:
            # z_i ~ Bern(theta) * N(0, 1/theta)
            # where theta = k/n
            # This gives Var(z_i) = 1 and E[||z||_0] = n*theta = k
            theta = config.k / n

            # Bernoulli: which features are active
            active_mask = torch.rand(n) < theta
            active_indices = torch.where(active_mask)[0]

            # Ensure at least one feature is active
            if len(active_indices) == 0:
                active_indices = torch.randint(0, n, (1,))

            if len(active_indices) > 0:
                # Gaussian with variance 1/theta
                std = (1.0 / theta) ** 0.5
                gaussian_coeffs = torch.randn(len(active_indices)) * std

                if config.positive_only:
                    coefficients[i, active_indices] = torch.abs(gaussian_coeffs)
                else:
                    coefficients[i, active_indices] = gaussian_coeffs

        else:
            raise ValueError(f"Unknown sparsity_mode: {config.sparsity_mode}")

    # Compute representations as linear combinations
    representations = coefficients @ features

    return representations, coefficients
