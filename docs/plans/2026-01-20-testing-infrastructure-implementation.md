# Testing Infrastructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reusable testing framework to validate monosemantic feature extraction on synthetic data with known ground truth.

**Architecture:** Modular Python + PyTorch design with separate modules for configuration, synthetic data generation, feature extraction algorithm, and evaluation metrics. TDD approach starting with the simplest case (ε=0 orthogonal features).

**Tech Stack:** Python 3.11+, PyTorch, pytest

---

## Task 0: Project Setup

**Files:**
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `tests/__init__.py`
- Create: `experiments/__init__.py`
- Create: `pyproject.toml`

**Step 1: Create project structure**

```bash
mkdir -p src tests experiments results
touch src/__init__.py tests/__init__.py experiments/__init__.py
```

**Step 2: Create pyproject.toml**

Create `pyproject.toml`:

```toml
[project]
name = "monosemantic-features"
version = "0.1.0"
description = "Testing infrastructure for monosemantic feature extraction"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]
```

**Step 3: Install dependencies**

```bash
pip install -e ".[dev]"
```

**Step 4: Verify installation**

```bash
python -c "import torch; print(torch.__version__)"
pytest --version
```

**Step 5: Commit**

```bash
git add .
git commit -m "chore: initialize project structure with PyTorch dependencies"
```

---

## Task 1: Configuration Dataclasses

**Files:**
- Create: `src/config.py`
- Create: `tests/test_config.py`

**Step 1: Write failing test for SyntheticConfig**

Create `tests/test_config.py`:

```python
import pytest
from src.config import SyntheticConfig, compute_coef_min


def test_synthetic_config_defaults():
    config = SyntheticConfig()
    assert config.d == 64
    assert config.n == 64
    assert config.epsilon == 0.0
    assert config.num_representations == 1000
    assert config.sparsity_mode == "fixed"
    assert config.k == 3


def test_compute_coef_min_with_epsilon_zero():
    config = SyntheticConfig(epsilon=0.0, k=3, coef_factor=10.0, coef_min_floor=0.1)
    # When epsilon=0, coef_min should be coef_min_floor
    assert compute_coef_min(config) == 0.1


def test_compute_coef_min_with_epsilon_nonzero():
    config = SyntheticConfig(epsilon=0.1, k=3, coef_factor=10.0, coef_min_floor=0.1)
    # coef_min = max(10 * 3 * 0.1, 0.1) = max(3.0, 0.1) = 3.0
    assert compute_coef_min(config) == 3.0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'src.config'"

**Step 3: Write implementation**

Create `src/config.py`:

```python
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_config.py -v
```

Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add configuration dataclasses"
```

---

## Task 2: Feature Basis Generation (ε=0 orthogonal case)

**Files:**
- Create: `src/synthetic.py`
- Create: `tests/test_synthetic.py`

**Step 1: Write failing test for orthogonal basis**

Create `tests/test_synthetic.py`:

```python
import pytest
import torch
from src.synthetic import generate_feature_basis


def test_generate_orthogonal_basis_shape():
    features = generate_feature_basis(d=64, n=64, epsilon=0.0)
    assert features.shape == (64, 64)


def test_generate_orthogonal_basis_unit_norm():
    features = generate_feature_basis(d=64, n=64, epsilon=0.0)
    norms = torch.norm(features, dim=1)
    assert torch.allclose(norms, torch.ones(64), atol=1e-6)


def test_generate_orthogonal_basis_orthogonality():
    features = generate_feature_basis(d=64, n=64, epsilon=0.0)
    # Gram matrix should be identity for orthonormal basis
    gram = features @ features.T
    identity = torch.eye(64)
    assert torch.allclose(gram, identity, atol=1e-6)


def test_generate_orthogonal_basis_requires_n_leq_d():
    with pytest.raises(ValueError, match="n <= d"):
        generate_feature_basis(d=32, n=64, epsilon=0.0)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_synthetic.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'src.synthetic'"

**Step 3: Write implementation**

Create `src/synthetic.py`:

```python
import torch


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
        # Generate random matrix and orthogonalize via QR
        random_matrix = torch.randn(d, n)
        q, _ = torch.linalg.qr(random_matrix)
        # q is (d, n), we want (n, d)
        return q.T
    else:
        raise NotImplementedError(f"epsilon > 0 not yet implemented, got epsilon={epsilon}")
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_synthetic.py -v
```

Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/synthetic.py tests/test_synthetic.py
git commit -m "feat: add orthogonal feature basis generation"
```

---

## Task 3: Representation Generation

**Files:**
- Modify: `src/synthetic.py`
- Modify: `tests/test_synthetic.py`

**Step 1: Write failing tests for representation generation**

Append to `tests/test_synthetic.py`:

```python
from src.config import SyntheticConfig
from src.synthetic import generate_representations, generate_feature_basis


def test_generate_representations_shape():
    config = SyntheticConfig(d=64, n=64, epsilon=0.0, num_representations=100, k=3)
    features = generate_feature_basis(config.d, config.n, config.epsilon)
    representations, coefficients = generate_representations(features, config)

    assert representations.shape == (100, 64)
    assert coefficients.shape == (100, 64)


def test_generate_representations_fixed_sparsity():
    config = SyntheticConfig(
        d=64, n=64, epsilon=0.0, num_representations=100,
        sparsity_mode="fixed", k=3
    )
    features = generate_feature_basis(config.d, config.n, config.epsilon)
    _, coefficients = generate_representations(features, config)

    # Each representation should have exactly k=3 non-zero coefficients
    nonzero_counts = (coefficients != 0).sum(dim=1)
    assert torch.all(nonzero_counts == 3)


def test_generate_representations_coefficient_bounds():
    config = SyntheticConfig(
        d=64, n=64, epsilon=0.0, num_representations=100,
        k=3, coef_min_floor=0.1, coef_max=1.0
    )
    features = generate_feature_basis(config.d, config.n, config.epsilon)
    _, coefficients = generate_representations(features, config)

    # Non-zero coefficients should have |c| in [0.1, 1.0]
    nonzero_mask = coefficients != 0
    nonzero_abs = torch.abs(coefficients[nonzero_mask])
    assert torch.all(nonzero_abs >= 0.1)
    assert torch.all(nonzero_abs <= 1.0)


def test_generate_representations_reconstruction():
    config = SyntheticConfig(d=64, n=64, epsilon=0.0, num_representations=100, k=3)
    features = generate_feature_basis(config.d, config.n, config.epsilon)
    representations, coefficients = generate_representations(features, config)

    # representations should equal coefficients @ features
    reconstructed = coefficients @ features
    assert torch.allclose(representations, reconstructed, atol=1e-6)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_synthetic.py::test_generate_representations_shape -v
```

Expected: FAIL with "cannot import name 'generate_representations'"

**Step 3: Write implementation**

Append to `src/synthetic.py`:

```python
from src.config import SyntheticConfig, compute_coef_min


def generate_representations(
    features: torch.Tensor,
    config: SyntheticConfig
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
    n, d = features.shape
    num_repr = config.num_representations
    coef_min = compute_coef_min(config)
    coef_max = config.coef_max

    # Initialize coefficients as zeros
    coefficients = torch.zeros(num_repr, n)

    for i in range(num_repr):
        # Determine which features are active based on sparsity mode
        if config.sparsity_mode == "fixed":
            active_indices = torch.randperm(n)[:config.k]
        elif config.sparsity_mode == "variable":
            k_min = config.k_min if config.k_min is not None else 1
            num_active = torch.randint(k_min, config.k + 1, (1,)).item()
            active_indices = torch.randperm(n)[:num_active]
        elif config.sparsity_mode == "probabilistic":
            p = config.k / n
            active_mask = torch.rand(n) < p
            active_indices = torch.where(active_mask)[0]
            # Ensure at least one feature is active
            if len(active_indices) == 0:
                active_indices = torch.randint(0, n, (1,))
        else:
            raise ValueError(f"Unknown sparsity_mode: {config.sparsity_mode}")

        # Generate coefficients for active features
        num_active = len(active_indices)
        # Uniform magnitude in [coef_min, coef_max]
        magnitudes = torch.rand(num_active) * (coef_max - coef_min) + coef_min
        # Random signs
        signs = torch.sign(torch.rand(num_active) - 0.5)
        signs[signs == 0] = 1  # Handle exact 0.5 case

        coefficients[i, active_indices] = magnitudes * signs

    # Compute representations as linear combinations
    representations = coefficients @ features

    return representations, coefficients
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_synthetic.py -v
```

Expected: PASS (8 tests)

**Step 5: Commit**

```bash
git add src/synthetic.py tests/test_synthetic.py
git commit -m "feat: add representation generation with configurable sparsity"
```

---

## Task 4: τ Threshold Derivation

**Files:**
- Modify: `src/extraction.py`
- Create: `tests/test_extraction.py`

**Step 1: Write failing tests for tau computation**

Create `tests/test_extraction.py`:

```python
import pytest
import torch
from src.config import SyntheticConfig, ExtractionConfig, compute_coef_min
from src.extraction import compute_tau_bounds, resolve_tau


def test_compute_tau_bounds_orthogonal():
    """For orthogonal features, tau bounds should be well-separated."""
    k = 3
    epsilon = 0.0
    coef_min = 0.1
    coef_max = 1.0

    tau_min, tau_max = compute_tau_bounds(k, epsilon, coef_min, coef_max)

    # tau_max should be less than tau_min for valid separation
    assert tau_max < tau_min


def test_resolve_tau_auto():
    """When tau is None, it should be auto-derived."""
    synthetic = SyntheticConfig(epsilon=0.0, k=3, coef_min_floor=0.1)
    extraction = ExtractionConfig(tau=None, tau_margin=0.5)

    tau = resolve_tau(extraction, synthetic)

    assert tau > 0
    assert tau < 1


def test_resolve_tau_manual():
    """When tau is set, it should be used directly."""
    synthetic = SyntheticConfig(epsilon=0.0, k=3)
    extraction = ExtractionConfig(tau=0.7)

    tau = resolve_tau(extraction, synthetic)

    assert tau == 0.7
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_extraction.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'src.extraction'"

**Step 3: Write implementation**

Create `src/extraction.py`:

```python
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_extraction.py -v
```

Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/extraction.py tests/test_extraction.py
git commit -m "feat: add tau threshold derivation"
```

---

## Task 5: Neighbor Finding

**Files:**
- Modify: `src/extraction.py`
- Modify: `tests/test_extraction.py`

**Step 1: Write failing tests for neighbor finding**

Append to `tests/test_extraction.py`:

```python
from src.extraction import find_neighbors


def test_find_neighbors_includes_self():
    """A representation should always be its own neighbor."""
    representations = torch.eye(4)  # 4 orthogonal unit vectors
    neighbors = find_neighbors(representations, target_idx=0, tau=0.5)
    assert 0 in neighbors


def test_find_neighbors_orthogonal():
    """Orthogonal representations shouldn't be neighbors (except self)."""
    representations = torch.eye(4)
    neighbors = find_neighbors(representations, target_idx=0, tau=0.5)
    assert len(neighbors) == 1
    assert neighbors[0] == 0


def test_find_neighbors_similar():
    """Similar representations should be neighbors."""
    representations = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],  # Similar to first
        [0.0, 1.0, 0.0],  # Orthogonal to first
    ])
    # Normalize
    representations = representations / representations.norm(dim=1, keepdim=True)

    neighbors = find_neighbors(representations, target_idx=0, tau=0.8)
    assert 0 in neighbors
    assert 1 in neighbors
    assert 2 not in neighbors
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_extraction.py::test_find_neighbors_includes_self -v
```

Expected: FAIL with "cannot import name 'find_neighbors'"

**Step 3: Write implementation**

Append to `src/extraction.py`:

```python
def find_neighbors(
    representations: torch.Tensor,
    target_idx: int,
    tau: float
) -> torch.Tensor:
    """
    Find indices of representations with cosine similarity ≥ τ to target.

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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_extraction.py -v
```

Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add src/extraction.py tests/test_extraction.py
git commit -m "feat: add neighbor finding by cosine similarity"
```

---

## Task 6: Clustering by Neighbor Sets

**Files:**
- Modify: `src/extraction.py`
- Modify: `tests/test_extraction.py`

**Step 1: Write failing tests for clustering**

Append to `tests/test_extraction.py`:

```python
from src.extraction import cluster_by_neighbors


def test_cluster_by_neighbors_distinct_features():
    """Representations with same dominant feature should cluster together."""
    # Create representations: 3 dominated by f1, 2 dominated by f2
    f1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    f2 = torch.tensor([0.0, 1.0, 0.0, 0.0])

    representations = torch.stack([
        f1 + 0.1 * torch.randn(4),  # 0: dominated by f1
        f1 + 0.1 * torch.randn(4),  # 1: dominated by f1
        f1 + 0.1 * torch.randn(4),  # 2: dominated by f1
        f2 + 0.1 * torch.randn(4),  # 3: dominated by f2
        f2 + 0.1 * torch.randn(4),  # 4: dominated by f2
    ])
    representations = representations / representations.norm(dim=1, keepdim=True)

    clusters = cluster_by_neighbors(representations, tau=0.8)

    # Should have 2 clusters
    assert len(clusters) == 2


def test_cluster_by_neighbors_returns_dict():
    representations = torch.eye(3)
    clusters = cluster_by_neighbors(representations, tau=0.5)

    # Each orthogonal vector forms its own cluster
    assert isinstance(clusters, dict)
    assert len(clusters) == 3
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_extraction.py::test_cluster_by_neighbors_returns_dict -v
```

Expected: FAIL with "cannot import name 'cluster_by_neighbors'"

**Step 3: Write implementation**

Append to `src/extraction.py`:

```python
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_extraction.py -v
```

Expected: PASS (8 tests)

**Step 5: Commit**

```bash
git add src/extraction.py tests/test_extraction.py
git commit -m "feat: add neighbor-based clustering"
```

---

## Task 7: Nullspace Computation

**Files:**
- Modify: `src/extraction.py`
- Modify: `tests/test_extraction.py`

**Step 1: Write failing tests for nullspace**

Append to `tests/test_extraction.py`:

```python
from src.extraction import compute_nullspace


def test_compute_nullspace_orthogonal():
    """Nullspace of non-neighbors should contain the shared feature direction."""
    # 4D space, 3 representations along first 3 axes
    representations = torch.eye(4)[:3]  # (3, 4)

    # Target is [0], neighbors are [0], non-neighbors are [1, 2]
    neighbor_indices = torch.tensor([0])

    # Nullspace of representations[1] and [2] (which span e2, e3)
    # should include directions e1 and e4
    nullspace = compute_nullspace(representations, neighbor_indices, epsilon=0.0)

    # Nullspace should have rank 2 (4D space minus 2 non-neighbor directions)
    assert nullspace.shape[0] == 2


def test_compute_nullspace_contains_shared_direction():
    """The nullspace should contain the direction shared by neighbors."""
    # Create representations where neighbors share feature f1
    f1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    f2 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    f3 = torch.tensor([0.0, 0.0, 1.0, 0.0])

    representations = torch.stack([
        f1,  # 0: neighbor (has f1)
        f1,  # 1: neighbor (has f1)
        f2,  # 2: non-neighbor
        f3,  # 3: non-neighbor
    ])

    neighbor_indices = torch.tensor([0, 1])
    nullspace = compute_nullspace(representations, neighbor_indices, epsilon=0.0)

    # Project f1 onto nullspace - should have high component
    projection = nullspace @ f1
    assert projection.norm() > 0.9  # f1 should be mostly in nullspace
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_extraction.py::test_compute_nullspace_orthogonal -v
```

Expected: FAIL with "cannot import name 'compute_nullspace'"

**Step 3: Write implementation**

Append to `src/extraction.py`:

```python
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_extraction.py -v
```

Expected: PASS (10 tests)

**Step 5: Commit**

```bash
git add src/extraction.py tests/test_extraction.py
git commit -m "feat: add nullspace computation via SVD"
```

---

## Task 8: Feature Extraction

**Files:**
- Modify: `src/extraction.py`
- Modify: `tests/test_extraction.py`

**Step 1: Write failing tests for feature extraction**

Append to `tests/test_extraction.py`:

```python
from src.extraction import extract_feature


def test_extract_feature_recovers_shared_direction():
    """Extract feature should recover the direction shared by neighbors."""
    # Create representations where neighbors share feature f1
    f1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    f2 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    f3 = torch.tensor([0.0, 0.0, 1.0, 0.0])

    representations = torch.stack([
        f1,  # 0: neighbor
        f1,  # 1: neighbor
        f2,  # 2: non-neighbor
        f3,  # 3: non-neighbor
    ])

    neighbor_indices = torch.tensor([0, 1])

    nullspace = compute_nullspace(representations, neighbor_indices, epsilon=0.0)
    extracted = extract_feature(representations, neighbor_indices, nullspace)

    # Extracted feature should align with f1
    alignment = torch.abs(extracted @ f1)
    assert alignment > 0.99


def test_extract_feature_unit_norm():
    """Extracted feature should have unit norm."""
    representations = torch.randn(10, 8)
    representations = representations / representations.norm(dim=1, keepdim=True)
    neighbor_indices = torch.tensor([0, 1, 2])

    nullspace = compute_nullspace(representations, neighbor_indices, epsilon=0.0)
    if nullspace.shape[0] > 0:  # Only if nullspace is non-trivial
        extracted = extract_feature(representations, neighbor_indices, nullspace)
        assert torch.isclose(extracted.norm(), torch.tensor(1.0), atol=1e-6)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_extraction.py::test_extract_feature_recovers_shared_direction -v
```

Expected: FAIL with "cannot import name 'extract_feature'"

**Step 3: Write implementation**

Append to `src/extraction.py`:

```python
def extract_feature(
    representations: torch.Tensor,
    neighbor_indices: torch.Tensor,
    nullspace: torch.Tensor
) -> torch.Tensor:
    """
    Extract feature by projecting neighbors onto nullspace and finding dominant direction.

    Args:
        representations: (num_repr, d) tensor
        neighbor_indices: 1D tensor of neighbor indices
        nullspace: (k, d) tensor of nullspace basis vectors

    Returns:
        (d,) unit-norm feature vector
    """
    if nullspace.shape[0] == 0:
        raise ValueError("Nullspace is empty, cannot extract feature")

    # Get neighbor representations
    neighbors = representations[neighbor_indices]  # (m, d)

    # Average the neighbors
    avg_neighbor = neighbors.mean(dim=0)  # (d,)

    # Project onto nullspace: project = nullspace.T @ nullspace @ avg_neighbor
    # But we want the component in the nullspace
    projection = nullspace.T @ (nullspace @ avg_neighbor)  # (d,)

    # Normalize to unit vector
    norm = projection.norm()
    if norm < 1e-8:
        raise ValueError("Projection onto nullspace has zero norm")

    feature = projection / norm

    return feature
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_extraction.py -v
```

Expected: PASS (12 tests)

**Step 5: Commit**

```bash
git add src/extraction.py tests/test_extraction.py
git commit -m "feat: add single feature extraction via nullspace projection"
```

---

## Task 9: Full Feature Extraction Pipeline

**Files:**
- Modify: `src/extraction.py`
- Modify: `tests/test_extraction.py`

**Step 1: Write failing tests for full pipeline**

Append to `tests/test_extraction.py`:

```python
from src.synthetic import generate_feature_basis, generate_representations
from src.extraction import extract_all_features


def test_extract_all_features_orthogonal_basis():
    """Full pipeline should recover features from orthogonal basis."""
    torch.manual_seed(42)

    config = SyntheticConfig(
        d=16, n=16, epsilon=0.0, num_representations=200,
        sparsity_mode="fixed", k=2, coef_min_floor=0.3
    )
    extraction_config = ExtractionConfig(tau=None, tau_margin=0.5, epsilon=0.0)

    features = generate_feature_basis(config.d, config.n, config.epsilon)
    representations, _ = generate_representations(features, config)

    extracted = extract_all_features(representations, extraction_config, config)

    # Should extract some features (at least a few)
    assert extracted.shape[0] > 0
    assert extracted.shape[1] == config.d


def test_extract_all_features_unit_norm():
    """All extracted features should have unit norm."""
    torch.manual_seed(42)

    config = SyntheticConfig(
        d=16, n=16, epsilon=0.0, num_representations=200,
        sparsity_mode="fixed", k=2, coef_min_floor=0.3
    )
    extraction_config = ExtractionConfig(tau=None, epsilon=0.0)

    features = generate_feature_basis(config.d, config.n, config.epsilon)
    representations, _ = generate_representations(features, config)

    extracted = extract_all_features(representations, extraction_config, config)

    norms = torch.norm(extracted, dim=1)
    assert torch.allclose(norms, torch.ones(extracted.shape[0]), atol=1e-6)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_extraction.py::test_extract_all_features_orthogonal_basis -v
```

Expected: FAIL with "cannot import name 'extract_all_features'"

**Step 3: Write implementation**

Append to `src/extraction.py`:

```python
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_extraction.py -v
```

Expected: PASS (14 tests)

**Step 5: Commit**

```bash
git add src/extraction.py tests/test_extraction.py
git commit -m "feat: add full feature extraction pipeline with deduplication"
```

---

## Task 10: Metrics - Feature Matching

**Files:**
- Create: `src/metrics.py`
- Create: `tests/test_metrics.py`

**Step 1: Write failing tests for feature matching**

Create `tests/test_metrics.py`:

```python
import pytest
import torch
from src.metrics import match_features


def test_match_features_perfect_recovery():
    """Identical features should match perfectly."""
    ground_truth = torch.eye(4)
    extracted = torch.eye(4)

    matching, scores = match_features(extracted, ground_truth, threshold=0.9)

    assert len(matching) == 4
    assert torch.all(scores > 0.99)


def test_match_features_partial_recovery():
    """Should only match features above threshold."""
    ground_truth = torch.eye(4)
    extracted = torch.eye(4)[:2]  # Only first 2 features

    matching, scores = match_features(extracted, ground_truth, threshold=0.9)

    assert len(matching) == 2


def test_match_features_sign_invariant():
    """Matching should be invariant to sign flips."""
    ground_truth = torch.eye(4)
    extracted = -torch.eye(4)  # Negated

    matching, scores = match_features(extracted, ground_truth, threshold=0.9)

    assert len(matching) == 4
    assert torch.all(scores > 0.99)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_metrics.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'src.metrics'"

**Step 3: Write implementation**

Create `src/metrics.py`:

```python
import torch
from dataclasses import dataclass


def match_features(
    extracted: torch.Tensor,
    ground_truth: torch.Tensor,
    threshold: float = 0.9
) -> tuple[dict[int, int], torch.Tensor]:
    """
    Match extracted features to ground truth using greedy best-match.

    Uses absolute cosine similarity to handle sign flips.

    Args:
        extracted: (m, d) tensor of extracted features
        ground_truth: (n, d) tensor of ground truth features
        threshold: Minimum cosine similarity to count as match

    Returns:
        matching: {extracted_idx: ground_truth_idx}
        scores: (m,) tensor of alignment scores for each extracted feature
    """
    m = extracted.shape[0]
    n = ground_truth.shape[0]

    if m == 0:
        return {}, torch.tensor([])

    # Compute pairwise absolute cosine similarities
    # Assuming unit norm features
    sims = torch.abs(extracted @ ground_truth.T)  # (m, n)

    matching = {}
    scores = torch.zeros(m)
    used_gt = set()

    # Greedy matching: repeatedly match highest similarity pair
    for _ in range(min(m, n)):
        # Mask out already used ground truth features
        masked_sims = sims.clone()
        for gt_idx in used_gt:
            masked_sims[:, gt_idx] = -1
        for ext_idx in matching:
            masked_sims[ext_idx, :] = -1

        # Find best remaining match
        best_sim = masked_sims.max()
        if best_sim < threshold:
            break

        best_idx = (masked_sims == best_sim).nonzero()[0]
        ext_idx, gt_idx = best_idx[0].item(), best_idx[1].item()

        matching[ext_idx] = gt_idx
        scores[ext_idx] = best_sim
        used_gt.add(gt_idx)

    return matching, scores
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_metrics.py -v
```

Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: add feature matching with sign invariance"
```

---

## Task 11: Metrics - Full Evaluation

**Files:**
- Modify: `src/metrics.py`
- Modify: `tests/test_metrics.py`

**Step 1: Write failing tests for full evaluation**

Append to `tests/test_metrics.py`:

```python
from src.metrics import evaluate, MetricsResult


def test_evaluate_perfect_recovery():
    """Perfect recovery should give recovery_rate=1.0."""
    ground_truth = torch.eye(4)
    extracted = torch.eye(4)
    representations = torch.eye(4)  # Each representation is one feature
    coefficients = torch.eye(4)  # Each has coefficient 1 on one feature

    result = evaluate(extracted, ground_truth, representations, coefficients)

    assert isinstance(result, MetricsResult)
    assert result.recovery_rate == 1.0
    assert result.mean_alignment > 0.99


def test_evaluate_partial_recovery():
    """Partial recovery should give recovery_rate < 1.0."""
    ground_truth = torch.eye(4)
    extracted = torch.eye(4)[:2]  # Only 2 features recovered
    representations = torch.eye(4)
    coefficients = torch.eye(4)

    result = evaluate(extracted, ground_truth, representations, coefficients)

    assert result.recovery_rate == 0.5  # 2 out of 4


def test_evaluate_returns_diagnostics():
    """Evaluation should return matching diagnostics."""
    ground_truth = torch.eye(4)
    extracted = torch.eye(4)[:3]
    representations = torch.eye(4)
    coefficients = torch.eye(4)

    result = evaluate(extracted, ground_truth, representations, coefficients)

    assert len(result.feature_matching) == 3
    assert len(result.unmatched_true) == 1
    assert len(result.unmatched_extracted) == 0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_metrics.py::test_evaluate_perfect_recovery -v
```

Expected: FAIL with "cannot import name 'evaluate'"

**Step 3: Write implementation**

Append to `src/metrics.py`:

```python
@dataclass
class MetricsResult:
    """Results from feature extraction evaluation."""
    recovery_rate: float              # Fraction of true features recovered
    alignment_scores: torch.Tensor    # Per-feature cosine similarity (best match)
    mean_alignment: float             # Average alignment across recovered features
    reconstruction_error: float       # Mean squared error of reconstruction

    # Diagnostics
    feature_matching: dict[int, int]  # Maps extracted idx → ground truth idx
    unmatched_true: list[int]         # Ground truth features not recovered
    unmatched_extracted: list[int]    # Extracted features matching nothing


def compute_reconstruction_error(
    representations: torch.Tensor,
    true_coefficients: torch.Tensor,
    extracted_features: torch.Tensor,
    feature_matching: dict[int, int],
    ground_truth_features: torch.Tensor
) -> float:
    """
    Compute reconstruction error using extracted features.

    Re-estimates coefficients by projecting representations onto extracted features,
    then computes MSE against original representations.

    Args:
        representations: (num_repr, d) original representations
        true_coefficients: (num_repr, n) ground truth sparse coefficients
        extracted_features: (m, d) extracted features
        feature_matching: mapping from extracted to ground truth indices
        ground_truth_features: (n, d) ground truth features

    Returns:
        Mean squared error of reconstruction
    """
    if extracted_features.shape[0] == 0:
        return float('inf')

    # Project representations onto extracted features to get estimated coefficients
    # For unit-norm features: coef = repr @ feature
    estimated_coefs = representations @ extracted_features.T  # (num_repr, m)

    # Reconstruct using extracted features
    reconstructed = estimated_coefs @ extracted_features  # (num_repr, d)

    # Compute MSE
    mse = ((representations - reconstructed) ** 2).mean().item()

    return mse


def evaluate(
    extracted: torch.Tensor,
    ground_truth: torch.Tensor,
    representations: torch.Tensor,
    true_coefficients: torch.Tensor,
    match_threshold: float = 0.9
) -> MetricsResult:
    """
    Compute all evaluation metrics.

    Args:
        extracted: (m, d) extracted features
        ground_truth: (n, d) ground truth features
        representations: (num_repr, d) representations
        true_coefficients: (num_repr, n) ground truth coefficients
        match_threshold: minimum cosine similarity to count as recovered

    Returns:
        MetricsResult with all metrics and diagnostics
    """
    n = ground_truth.shape[0]
    m = extracted.shape[0]

    # Match features
    matching, scores = match_features(extracted, ground_truth, match_threshold)

    # Recovery rate
    recovery_rate = len(matching) / n if n > 0 else 0.0

    # Mean alignment (only for matched features)
    matched_scores = scores[scores > 0]
    mean_alignment = matched_scores.mean().item() if len(matched_scores) > 0 else 0.0

    # Reconstruction error
    reconstruction_error = compute_reconstruction_error(
        representations, true_coefficients, extracted, matching, ground_truth
    )

    # Diagnostics
    matched_gt = set(matching.values())
    unmatched_true = [i for i in range(n) if i not in matched_gt]
    unmatched_extracted = [i for i in range(m) if i not in matching]

    return MetricsResult(
        recovery_rate=recovery_rate,
        alignment_scores=scores,
        mean_alignment=mean_alignment,
        reconstruction_error=reconstruction_error,
        feature_matching=matching,
        unmatched_true=unmatched_true,
        unmatched_extracted=unmatched_extracted,
    )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_metrics.py -v
```

Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: add full evaluation metrics with diagnostics"
```

---

## Task 12: Experiment Runner

**Files:**
- Create: `experiments/run_experiment.py`
- Modify: `tests/test_extraction.py`

**Step 1: Write failing test for experiment runner**

Create `tests/test_experiment.py`:

```python
import pytest
import torch
from src.config import SyntheticConfig, ExtractionConfig, ExperimentConfig
from experiments.run_experiment import run_experiment


def test_run_experiment_orthogonal():
    """Full experiment should run and return metrics."""
    config = ExperimentConfig(
        synthetic=SyntheticConfig(
            d=16, n=16, epsilon=0.0, num_representations=200,
            sparsity_mode="fixed", k=2, coef_min_floor=0.3
        ),
        extraction=ExtractionConfig(tau=None, epsilon=0.0),
        seed=42,
        match_threshold=0.9
    )

    result = run_experiment(config)

    assert result.recovery_rate >= 0
    assert result.recovery_rate <= 1
    assert result.mean_alignment >= 0


def test_run_experiment_deterministic():
    """Same seed should give same results."""
    config = ExperimentConfig(
        synthetic=SyntheticConfig(
            d=16, n=16, epsilon=0.0, num_representations=100,
            sparsity_mode="fixed", k=2
        ),
        extraction=ExtractionConfig(tau=None, epsilon=0.0),
        seed=123
    )

    result1 = run_experiment(config)
    result2 = run_experiment(config)

    assert result1.recovery_rate == result2.recovery_rate
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_experiment.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

Create `experiments/run_experiment.py`:

```python
import torch
import json
from pathlib import Path
from datetime import datetime

from src.config import ExperimentConfig, SyntheticConfig, ExtractionConfig
from src.synthetic import generate_feature_basis, generate_representations
from src.extraction import extract_all_features
from src.metrics import evaluate, MetricsResult


def run_experiment(config: ExperimentConfig) -> MetricsResult:
    """
    Execute full experiment pipeline.

    1. Set random seed for reproducibility
    2. Generate feature basis
    3. Generate representations with known coefficients
    4. Run extraction algorithm
    5. Evaluate metrics against ground truth
    6. Return results

    Args:
        config: ExperimentConfig with all settings

    Returns:
        MetricsResult with evaluation metrics
    """
    # Set seed
    torch.manual_seed(config.seed)

    # Generate ground truth
    features = generate_feature_basis(
        config.synthetic.d,
        config.synthetic.n,
        config.synthetic.epsilon
    )

    representations, coefficients = generate_representations(
        features, config.synthetic
    )

    # Run extraction
    extracted = extract_all_features(
        representations, config.extraction, config.synthetic
    )

    # Evaluate
    result = evaluate(
        extracted=extracted,
        ground_truth=features,
        representations=representations,
        true_coefficients=coefficients,
        match_threshold=config.match_threshold
    )

    return result


def save_results(result: MetricsResult, config: ExperimentConfig, path: Path) -> None:
    """Save experiment results to JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "synthetic": {
                "d": config.synthetic.d,
                "n": config.synthetic.n,
                "epsilon": config.synthetic.epsilon,
                "num_representations": config.synthetic.num_representations,
                "sparsity_mode": config.synthetic.sparsity_mode,
                "k": config.synthetic.k,
            },
            "extraction": {
                "tau": config.extraction.tau,
                "epsilon": config.extraction.epsilon,
            },
            "seed": config.seed,
        },
        "results": {
            "recovery_rate": result.recovery_rate,
            "mean_alignment": result.mean_alignment,
            "reconstruction_error": result.reconstruction_error,
            "num_extracted": len(result.feature_matching),
            "num_unmatched_true": len(result.unmatched_true),
            "num_unmatched_extracted": len(result.unmatched_extracted),
        }
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run monosemantic feature extraction experiment")
    parser.add_argument("--d", type=int, default=64, help="Representation dimension")
    parser.add_argument("--n", type=int, default=64, help="Number of features")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Orthogonality tolerance")
    parser.add_argument("--num-repr", type=int, default=1000, help="Number of representations")
    parser.add_argument("--k", type=int, default=3, help="Sparsity (active features)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output path for results JSON")

    args = parser.parse_args()

    config = ExperimentConfig(
        synthetic=SyntheticConfig(
            d=args.d,
            n=args.n,
            epsilon=args.epsilon,
            num_representations=args.num_repr,
            k=args.k,
        ),
        extraction=ExtractionConfig(tau=None, epsilon=args.epsilon),
        seed=args.seed,
    )

    print(f"Running experiment: d={args.d}, n={args.n}, epsilon={args.epsilon}, k={args.k}")
    result = run_experiment(config)

    print(f"\nResults:")
    print(f"  Recovery rate: {result.recovery_rate:.2%}")
    print(f"  Mean alignment: {result.mean_alignment:.4f}")
    print(f"  Reconstruction error: {result.reconstruction_error:.6f}")
    print(f"  Features extracted: {len(result.feature_matching)}")
    print(f"  Unmatched true: {len(result.unmatched_true)}")
    print(f"  Unmatched extracted: {len(result.unmatched_extracted)}")

    if args.output:
        save_results(result, config, Path(args.output))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_experiment.py -v
```

Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add experiments/run_experiment.py tests/test_experiment.py
git commit -m "feat: add experiment runner with CLI"
```

---

## Task 13: Integration Test - Orthogonal Recovery

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

Create `tests/test_integration.py`:

```python
"""
Integration tests validating the full pipeline on synthetic data.
"""
import pytest
import torch
from src.config import SyntheticConfig, ExtractionConfig, ExperimentConfig
from experiments.run_experiment import run_experiment


class TestOrthogonalRecovery:
    """Test feature recovery with orthogonal (ε=0) feature basis."""

    def test_high_recovery_rate(self):
        """Should recover most features with good parameters."""
        config = ExperimentConfig(
            synthetic=SyntheticConfig(
                d=32, n=32, epsilon=0.0, num_representations=500,
                sparsity_mode="fixed", k=2, coef_min_floor=0.3
            ),
            extraction=ExtractionConfig(tau=None, epsilon=0.0),
            seed=42,
            match_threshold=0.9
        )

        result = run_experiment(config)

        # Should recover at least 50% of features
        assert result.recovery_rate > 0.5, f"Recovery rate {result.recovery_rate} too low"

    def test_high_alignment(self):
        """Recovered features should closely match ground truth."""
        config = ExperimentConfig(
            synthetic=SyntheticConfig(
                d=32, n=32, epsilon=0.0, num_representations=500,
                sparsity_mode="fixed", k=2, coef_min_floor=0.3
            ),
            extraction=ExtractionConfig(tau=None, epsilon=0.0),
            seed=42,
            match_threshold=0.9
        )

        result = run_experiment(config)

        # Matched features should have high alignment
        assert result.mean_alignment > 0.95, f"Mean alignment {result.mean_alignment} too low"

    def test_low_reconstruction_error(self):
        """Should reconstruct representations with low error."""
        config = ExperimentConfig(
            synthetic=SyntheticConfig(
                d=32, n=32, epsilon=0.0, num_representations=500,
                sparsity_mode="fixed", k=2, coef_min_floor=0.3
            ),
            extraction=ExtractionConfig(tau=None, epsilon=0.0),
            seed=42,
        )

        result = run_experiment(config)

        # Reconstruction error should be reasonable
        assert result.reconstruction_error < 1.0, f"Reconstruction error {result.reconstruction_error} too high"

    def test_variable_sparsity(self):
        """Should work with variable sparsity mode."""
        config = ExperimentConfig(
            synthetic=SyntheticConfig(
                d=32, n=32, epsilon=0.0, num_representations=500,
                sparsity_mode="variable", k=4, k_min=1, coef_min_floor=0.3
            ),
            extraction=ExtractionConfig(tau=None, epsilon=0.0),
            seed=42,
        )

        result = run_experiment(config)

        # Should still recover features
        assert result.recovery_rate > 0.3

    def test_probabilistic_sparsity(self):
        """Should work with probabilistic sparsity mode."""
        config = ExperimentConfig(
            synthetic=SyntheticConfig(
                d=32, n=32, epsilon=0.0, num_representations=500,
                sparsity_mode="probabilistic", k=3, coef_min_floor=0.3
            ),
            extraction=ExtractionConfig(tau=None, epsilon=0.0),
            seed=42,
        )

        result = run_experiment(config)

        # Should still recover features
        assert result.recovery_rate > 0.3
```

**Step 2: Run integration tests**

```bash
pytest tests/test_integration.py -v
```

Expected: PASS (5 tests)

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for orthogonal feature recovery"
```

---

## Task 14: Final Verification & Documentation

**Files:**
- Update: `README.md`

**Step 1: Run all tests**

```bash
pytest -v --tb=short
```

Expected: All tests PASS

**Step 2: Run a sample experiment**

```bash
python experiments/run_experiment.py --d 64 --n 64 --k 3 --num-repr 1000 --output results/test_run.json
```

Expected: Output showing recovery rate, alignment, reconstruction error

**Step 3: Update README**

Update `README.md` with usage instructions:

```markdown
# monosemantic-features

Testing infrastructure for validating monosemantic feature extraction methodology.

## Installation

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest -v
```

## Running Experiments

```bash
# Basic orthogonal experiment
python experiments/run_experiment.py --d 64 --n 64 --k 3

# Save results
python experiments/run_experiment.py --d 64 --n 64 --k 3 --output results/experiment.json
```

## Project Structure

- `src/config.py` - Configuration dataclasses
- `src/synthetic.py` - Feature basis and representation generation
- `src/extraction.py` - Feature extraction algorithm
- `src/metrics.py` - Evaluation metrics
- `experiments/run_experiment.py` - Experiment runner CLI
- `docs/plans/` - Design documents and implementation plans
- `writeup/` - CVPR 2025 paper

## Methodology

See `docs/plans/2026-01-20-testing-infrastructure-design.md` for detailed design.
```

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: update README with usage instructions"
```

**Step 5: Final push (optional)**

```bash
git push origin feature/testing-infrastructure
```

---

## Summary

| Task | Description | Tests |
|------|-------------|-------|
| 0 | Project setup | - |
| 1 | Configuration dataclasses | 3 |
| 2 | Orthogonal basis generation | 4 |
| 3 | Representation generation | 4 |
| 4 | τ threshold derivation | 3 |
| 5 | Neighbor finding | 3 |
| 6 | Neighbor-based clustering | 2 |
| 7 | Nullspace computation | 2 |
| 8 | Feature extraction | 2 |
| 9 | Full extraction pipeline | 2 |
| 10 | Feature matching | 3 |
| 11 | Full evaluation metrics | 3 |
| 12 | Experiment runner | 2 |
| 13 | Integration tests | 5 |
| 14 | Documentation | - |

**Total: 14 tasks, ~38 tests**
