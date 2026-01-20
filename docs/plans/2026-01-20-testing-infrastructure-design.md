# Testing Infrastructure for Monosemantic Feature Extraction

**Date:** 2026-01-20
**Purpose:** Validate the feature extraction methodology on synthetic data with known ground truth before applying to real LLM representations.

## Overview

Build a reusable testing framework that:
1. Generates synthetic representations from a known feature basis
2. Runs the extraction algorithm (τ-neighbors → nullspace → SVD)
3. Compares extracted features against ground truth using comprehensive metrics

Start with ε=0 (orthogonal features) to validate the basic algorithm, then increase ε to test ε-orthogonal bases.

## Module Structure

```
monosemantic-features/
├── src/
│   ├── __init__.py
│   ├── synthetic.py      # Feature basis & representation generation
│   ├── extraction.py     # τ-neighbors, ε-nullspace, SVD extraction
│   ├── metrics.py        # Recovery rate, alignment, reconstruction error
│   └── config.py         # Dataclasses for experiment configuration
├── tests/
│   └── test_orthogonal.py  # First validation: ε=0 case
├── experiments/
│   └── run_experiment.py   # CLI entry point for running experiments
└── results/                # Output directory for experiment results
```

## Configuration (`config.py`)

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class SyntheticConfig:
    d: int = 64                      # Representation dimension
    n: int = 64                      # Number of features in basis
    epsilon: float = 0.0             # Orthogonality tolerance (0 = orthogonal)
    num_representations: int = 1000  # How many to generate

    # Sparsity settings
    sparsity_mode: Literal["fixed", "variable", "probabilistic"] = "fixed"
    k: int = 3                       # Fixed: exactly k active
                                     # Variable: k is k_max (requires k_min)
                                     # Probabilistic: expected k active (p = k/n)
    k_min: int | None = None         # Only for variable mode

    # Coefficient settings
    coef_factor: float = 10.0        # Multiplier for minimum coefficient
    coef_max: float = 1.0            # Maximum |coefficient|
    coef_min_floor: float = 0.1      # Minimum |coefficient| regardless of epsilon

def compute_coef_min(config: SyntheticConfig) -> float:
    """coef_min = max(coef_factor * k * epsilon, coef_min_floor)"""
    return max(config.coef_factor * config.k * config.epsilon, config.coef_min_floor)


@dataclass
class ExtractionConfig:
    tau: float | None = None         # Cosine similarity threshold (None = auto-derive)
    tau_margin: float = 0.5          # Where between bounds: 0=tau_min, 1=tau_max
    epsilon: float = 0.0             # Noise threshold for nullspace


@dataclass
class ExperimentConfig:
    synthetic: SyntheticConfig
    extraction: ExtractionConfig
    seed: int = 42
    match_threshold: float = 0.9     # Min cosine sim to count feature as recovered
```

## Synthetic Data Generation (`synthetic.py`)

### Feature Basis Generator

```python
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
    For ε>0: Returns ε-orthogonal frame (allows n > d up to Welch bound)
    """
```

### Representation Generator

```python
def generate_representations(
    features: torch.Tensor,
    config: SyntheticConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate representations as sparse linear combinations of features.

    Coefficient generation:
    - Draw |c| uniformly from [coef_min, coef_max]
    - Assign random sign (+ or -)

    Sparsity modes:
    - fixed: Exactly k features active per representation
    - variable: Between k_min and k features active
    - probabilistic: Each feature active with probability k/n

    Returns:
        representations: (num_representations, d)
        coefficients: (num_representations, n) ground-truth sparse coefficients
    """
```

## Feature Extraction Algorithm (`extraction.py`)

### τ Threshold Derivation

```python
def compute_tau_bounds(
    k: int,
    epsilon: float,
    coef_min: float,
    coef_max: float
) -> tuple[float, float]:
    """
    Compute valid τ bounds based on generative parameters.

    Returns:
        tau_min: Worst-case similarity when sharing a feature
        tau_max: Best-case similarity when NOT sharing a feature

    Valid τ must satisfy tau_max < τ ≤ tau_min for clean separation.
    """

def resolve_tau(
    extraction_config: ExtractionConfig,
    synthetic_config: SyntheticConfig
) -> float:
    """
    If tau is None, compute from bounds using tau_margin.
    If tau is set manually, use it directly.
    """
```

### Core Extraction Functions

```python
def find_neighbors(
    representations: torch.Tensor,
    target_idx: int,
    tau: float
) -> torch.Tensor:
    """
    Find indices of representations with cosine similarity ≥ τ to target.
    Returns: 1D tensor of neighbor indices
    """

def cluster_by_neighbors(
    representations: torch.Tensor,
    tau: float
) -> dict[frozenset[int], list[int]]:
    """
    Group representation indices by their neighbor sets.
    Representations with identical neighbor sets share a dominant feature.

    Returns: {neighbor_set: [indices of representations with that neighbor set]}
    """

def compute_nullspace(
    representations: torch.Tensor,
    neighbor_indices: torch.Tensor,
    epsilon: float
) -> torch.Tensor:
    """
    Compute ε-nullspace of non-neighbor representations.
    Returns: (k, d) tensor of nullspace basis vectors
    """

def extract_feature(
    representations: torch.Tensor,
    neighbor_indices: torch.Tensor,
    nullspace: torch.Tensor
) -> torch.Tensor:
    """
    Project neighbors onto nullspace, then SVD to find dominant direction.
    Returns: (d,) unit-norm feature vector
    """

def extract_all_features(
    representations: torch.Tensor,
    config: ExtractionConfig
) -> torch.Tensor:
    """
    Main entry point for feature extraction.

    Algorithm:
    1. Cluster representations by identical neighbor sets
    2. For each cluster:
       a. Average the representations in the cluster
       b. Compute nullspace of non-neighbors
       c. Project averaged representation onto nullspace
       d. SVD to get dominant direction → extracted feature
    3. Return all extracted features

    Returns: (m, d) tensor of extracted features
    """
```

## Metrics (`metrics.py`)

```python
@dataclass
class MetricsResult:
    recovery_rate: float              # Fraction of true features recovered
    alignment_scores: torch.Tensor    # Per-feature cosine similarity (best match)
    mean_alignment: float             # Average alignment across recovered features
    reconstruction_error: float       # Mean squared error of reconstruction

    # Diagnostics
    feature_matching: dict[int, int]  # Maps extracted idx → ground truth idx
    unmatched_true: list[int]         # Ground truth features not recovered
    unmatched_extracted: list[int]    # Extracted features matching nothing


def match_features(
    extracted: torch.Tensor,
    ground_truth: torch.Tensor,
    threshold: float = 0.9
) -> tuple[dict[int, int], torch.Tensor]:
    """
    Match extracted features to ground truth using Hungarian algorithm.

    Returns:
        matching: {extracted_idx: ground_truth_idx}
        alignment_scores: cosine similarity for each match
    """

def compute_reconstruction_error(
    representations: torch.Tensor,
    true_coefficients: torch.Tensor,
    extracted_features: torch.Tensor,
    feature_matching: dict[int, int]
) -> float:
    """
    Re-estimate coefficients using extracted features, compute MSE vs original.
    """

def evaluate(
    extracted: torch.Tensor,
    ground_truth: torch.Tensor,
    representations: torch.Tensor,
    true_coefficients: torch.Tensor,
    match_threshold: float = 0.9
) -> MetricsResult:
    """
    Main entry point: compute all metrics.
    """
```

## Experiment Runner (`experiments/run_experiment.py`)

```python
def run_experiment(config: ExperimentConfig) -> MetricsResult:
    """
    Execute full experiment pipeline:
    1. Set random seed for reproducibility
    2. Generate feature basis
    3. Generate representations with known coefficients
    4. Run extraction algorithm
    5. Evaluate metrics against ground truth
    6. Return results
    """

def main():
    """
    CLI entry point:
    - Parse config from file or command line
    - Run experiment
    - Save results to results/ directory as JSON
    - Print summary to console
    """
```

## First Test Case: Orthogonal Features (ε=0)

```python
# tests/test_orthogonal.py

def test_orthogonal_recovery():
    """
    Simplest case: ε=0, orthogonal features, fixed sparsity.
    Algorithm should achieve ~100% recovery with high alignment.
    """
    config = ExperimentConfig(
        synthetic=SyntheticConfig(
            d=64,
            n=64,
            epsilon=0.0,
            num_representations=1000,
            sparsity_mode="fixed",
            k=3,
            coef_min_floor=0.1,
        ),
        extraction=ExtractionConfig(tau=None),
        seed=42,
        match_threshold=0.9
    )

    result = run_experiment(config)

    assert result.recovery_rate > 0.95
    assert result.mean_alignment > 0.99
```

## Future Extensions

1. **ε-orthogonal basis generation** - Implement algorithms to construct ε-orthogonal frames for n > d (up to Welch bound)
2. **Real representation integration** - Load actual LLM activations and run extraction
3. **Visualization** - Plot feature recovery vs ε, alignment distributions, etc.
4. **Hyperparameter sweeps** - Systematic exploration of τ, ε, k parameter space
