from dataclasses import dataclass
from typing import Literal


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    d: int = 64                          # Representation dimension
    n: int = 64                          # Number of features in basis
    # epsilon removed - now derived from generate_feature_basis
    num_representations: int = 1000      # How many to generate

    # Sparsity settings
    sparsity_mode: Literal["fixed", "variable", "probabilistic", "bernoulli_gaussian"] = "fixed"
    k: int = 3                           # Fixed: exactly k active
                                         # Variable: k is k_max (requires k_min)
                                         # Probabilistic/BG: expected k active (theta = k/n)
    k_min: int | None = None             # Only for variable mode

    # Coefficient settings (for non-BG modes)
    coef_factor: float = 10.0            # Multiplier for minimum coefficient
    coef_max: float = 1.0                # Maximum |coefficient|
    coef_min_floor: float = 0.1          # Minimum |coefficient| regardless of epsilon
    positive_only: bool = False          # If True, coefficients are always positive


def compute_coef_min(config: SyntheticConfig, epsilon: float = 0.0) -> float:
    """Compute minimum coefficient magnitude: max(factor * k * epsilon, floor)."""
    return max(config.coef_factor * config.k * epsilon, config.coef_min_floor)


@dataclass
class ExtractionConfig:
    """Configuration for feature extraction algorithm."""
    tau: float | None = None             # Dot product threshold (None = auto-derive)
    tau_margin: float = 0.5              # Where between bounds: 0=tau_lower, 1=tau_upper
    epsilon: float = 0.0                 # Noise threshold for nullspace
    max_neighbors: int | None = None     # Max size for neighbor sets (None = unlimited)
    neg_tau: float | None = None         # Max |cos_sim| for negative set (None = use complement)
    use_minimality_filter: bool = True   # Only extract from monosemantic targets


@dataclass
class ExperimentConfig:
    """Configuration for a complete experiment."""
    synthetic: SyntheticConfig
    extraction: ExtractionConfig
    seed: int = 42
    match_threshold: float = 0.9         # Min cosine sim to count feature as recovered
