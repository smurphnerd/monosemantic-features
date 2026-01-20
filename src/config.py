from dataclasses import dataclass
from typing import Literal


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    d: int = 64                          # Representation dimension
    n: int = 64                          # Number of features in basis
    epsilon: float = 0.0                 # Orthogonality tolerance (0 = orthogonal)
    num_representations: int = 1000      # How many to generate

    # Sparsity settings
    sparsity_mode: Literal["fixed", "variable", "probabilistic"] = "fixed"
    k: int = 3                           # Fixed: exactly k active
                                         # Variable: k is k_max (requires k_min)
                                         # Probabilistic: expected k active (p = k/n)
    k_min: int | None = None             # Only for variable mode

    # Coefficient settings
    coef_factor: float = 10.0            # Multiplier for minimum coefficient
    coef_max: float = 1.0                # Maximum |coefficient|
    coef_min_floor: float = 0.1          # Minimum |coefficient| regardless of epsilon


def compute_coef_min(config: SyntheticConfig) -> float:
    """Compute minimum coefficient magnitude: max(factor * k * epsilon, floor)."""
    return max(config.coef_factor * config.k * config.epsilon, config.coef_min_floor)


@dataclass
class ExtractionConfig:
    """Configuration for feature extraction algorithm."""
    tau: float | None = None             # Cosine similarity threshold (None = auto-derive)
    tau_margin: float = 0.5              # Where between bounds: 0=tau_min, 1=tau_max
    epsilon: float = 0.0                 # Noise threshold for nullspace


@dataclass
class ExperimentConfig:
    """Configuration for a complete experiment."""
    synthetic: SyntheticConfig
    extraction: ExtractionConfig
    seed: int = 42
    match_threshold: float = 0.9         # Min cosine sim to count feature as recovered
