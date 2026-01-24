import math
from dataclasses import dataclass

import torch

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


def generate_feature_basis(d: int, n: int, epsilon: float) -> torch.Tensor:
    """
    Generate n feature vectors in d dimensions that are ε-orthogonal.

    Args:
        d: Dimension of representation space
        n: Number of features
        epsilon: Orthogonality tolerance (|⟨f_i, f_j⟩| ≤ ε for i≠j)

    Returns:
        (n, d) tensor of unit-norm feature vectors

    For ε=0: Returns orthonormal basis (requires n ≤ d)
    For ε>0: Not yet implemented
    """
    if epsilon == 0.0:
        if n > d:
            raise ValueError(f"For epsilon=0, requires n <= d, got n={n}, d={d}")
        # QR decomposition produces orthonormal columns, so we create (d, n)
        # and transpose the result to get (n, d) where rows are feature vectors
        random_matrix = torch.randn(d, n)
        q, _ = torch.linalg.qr(random_matrix)
        return q.T  # (n, d) - rows are unit-norm orthogonal feature vectors
    else:
        raise NotImplementedError(
            f"epsilon > 0 not yet implemented, got epsilon={epsilon}"
        )


def generate_representations(
    features: torch.Tensor, config: SyntheticConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate representations as sparse linear combinations of features.

    Args:
        features: (n, d) tensor of feature vectors
        config: SyntheticConfig with sparsity and coefficient settings

    Returns:
        representations: (num_representations, d)
        coefficients: (num_representations, n) ground-truth sparse coefficients
    """
    n, _ = features.shape
    num_repr = config.num_representations
    coef_min = compute_coef_min(config)
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
