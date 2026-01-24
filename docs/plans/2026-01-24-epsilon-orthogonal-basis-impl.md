# Epsilon-Orthogonal Basis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate n unit-norm feature vectors in d dimensions with minimal pairwise coherence, supporting superposition research (n > d).

**Architecture:** For n ≤ d, use QR decomposition for orthonormal basis. For n > d, use Gram matrix alternating projection to minimize coherence toward Welch bound. Return a result object with achieved epsilon for downstream calibration.

**Tech Stack:** PyTorch (torch.linalg for eigendecomposition and QR)

---

### Task 1: Add welch_bound Function

**Files:**
- Modify: `src/synthetic.py`
- Test: `tests/test_synthetic.py`

**Step 1: Write the failing tests**

Add to `tests/test_synthetic.py`:

```python
def test_welch_bound_orthogonal_possible():
    """When n <= d, Welch bound is 0 (orthonormal basis possible)."""
    from src.synthetic import welch_bound
    assert welch_bound(n=5, d=10) == 0.0
    assert welch_bound(n=10, d=10) == 0.0


def test_welch_bound_overcomplete():
    """When n > d, Welch bound is positive."""
    from src.synthetic import welch_bound
    # n=100, d=64: sqrt((100-64)/(64*(100-1))) = sqrt(36/6336) ≈ 0.0754
    bound = welch_bound(n=100, d=64)
    assert bound > 0
    assert abs(bound - 0.0754) < 0.001


def test_welch_bound_formula():
    """Verify exact formula: sqrt((n-d)/(d*(n-1)))."""
    import math
    from src.synthetic import welch_bound
    n, d = 20, 10
    expected = math.sqrt((n - d) / (d * (n - 1)))
    assert abs(welch_bound(n, d) - expected) < 1e-10
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_synthetic.py::test_welch_bound_orthogonal_possible tests/test_synthetic.py::test_welch_bound_overcomplete tests/test_synthetic.py::test_welch_bound_formula -v`

Expected: FAIL with "cannot import name 'welch_bound'"

**Step 3: Write implementation**

Add to `src/synthetic.py` after imports:

```python
import math


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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_synthetic.py::test_welch_bound_orthogonal_possible tests/test_synthetic.py::test_welch_bound_overcomplete tests/test_synthetic.py::test_welch_bound_formula -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/synthetic.py tests/test_synthetic.py
git commit -m "feat: add welch_bound function for theoretical coherence minimum"
```

---

### Task 2: Add FeatureBasisResult Dataclass

**Files:**
- Modify: `src/synthetic.py`
- Test: `tests/test_synthetic.py`

**Step 1: Write the failing test**

Add to `tests/test_synthetic.py`:

```python
def test_feature_basis_result_dataclass():
    """FeatureBasisResult holds features and metadata."""
    import torch
    from src.synthetic import FeatureBasisResult

    features = torch.randn(5, 3)
    result = FeatureBasisResult(
        features=features,
        achieved_epsilon=0.1,
        welch_bound=0.05,
        converged=True
    )

    assert result.features.shape == (5, 3)
    assert result.achieved_epsilon == 0.1
    assert result.welch_bound == 0.05
    assert result.converged is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthetic.py::test_feature_basis_result_dataclass -v`

Expected: FAIL with "cannot import name 'FeatureBasisResult'"

**Step 3: Write implementation**

Add to `src/synthetic.py` after imports:

```python
from dataclasses import dataclass


@dataclass
class FeatureBasisResult:
    """Result of feature basis generation."""
    features: torch.Tensor      # (n, d) unit-norm feature vectors
    achieved_epsilon: float     # max |<f_i, f_j>| across all pairs
    welch_bound: float          # theoretical minimum for this n, d
    converged: bool             # True if iterative method stabilized
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthetic.py::test_feature_basis_result_dataclass -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/synthetic.py tests/test_synthetic.py
git commit -m "feat: add FeatureBasisResult dataclass"
```

---

### Task 3: Refactor generate_feature_basis for n <= d Case

**Files:**
- Modify: `src/synthetic.py`
- Modify: `tests/test_synthetic.py`

**Step 1: Write the failing test for new return type**

Add to `tests/test_synthetic.py`:

```python
def test_generate_feature_basis_returns_result():
    """generate_feature_basis returns FeatureBasisResult."""
    from src.synthetic import generate_feature_basis, FeatureBasisResult

    result = generate_feature_basis(d=5, n=3)

    assert isinstance(result, FeatureBasisResult)
    assert result.features.shape == (3, 5)
    assert result.achieved_epsilon == 0.0
    assert result.welch_bound == 0.0
    assert result.converged is True


def test_generate_feature_basis_orthonormal():
    """For n <= d, features are orthonormal."""
    import torch
    from src.synthetic import generate_feature_basis

    result = generate_feature_basis(d=5, n=5)
    F = result.features

    # Unit norm
    norms = torch.linalg.norm(F, dim=1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-6)

    # Orthogonal (G = I for orthonormal)
    G = F @ F.T
    assert torch.allclose(G, torch.eye(5), atol=1e-6)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_synthetic.py::test_generate_feature_basis_returns_result tests/test_synthetic.py::test_generate_feature_basis_orthonormal -v`

Expected: FAIL (old function returns Tensor, not FeatureBasisResult)

**Step 3: Update implementation**

Replace the existing `generate_feature_basis` function in `src/synthetic.py`:

```python
def generate_feature_basis(d: int, n: int) -> FeatureBasisResult:
    """
    Generate n unit-norm feature vectors in d dimensions with minimal coherence.

    Args:
        d: Dimension of representation space
        n: Number of features

    Returns:
        FeatureBasisResult with features and metadata

    For n <= d: Returns orthonormal basis (achieved_epsilon = 0)
    For n > d: Not yet implemented
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
        raise NotImplementedError(
            f"n > d not yet implemented, got n={n}, d={d}"
        )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_synthetic.py::test_generate_feature_basis_returns_result tests/test_synthetic.py::test_generate_feature_basis_orthonormal -v`

Expected: PASS

**Step 5: Fix existing tests that use old interface**

The existing tests use `generate_feature_basis(d, n, epsilon)` and expect a Tensor. Update them:

In `tests/test_synthetic.py`, find and update:

```python
# Old:
# features = generate_feature_basis(d, n, epsilon=0.0)

# New:
# result = generate_feature_basis(d, n)
# features = result.features
```

Update `test_generate_orthogonal_basis_shape`:
```python
def test_generate_orthogonal_basis_shape():
    result = generate_feature_basis(d=10, n=5)
    assert result.features.shape == (5, 10)
```

Update `test_generate_orthogonal_basis_unit_norm`:
```python
def test_generate_orthogonal_basis_unit_norm():
    result = generate_feature_basis(d=10, n=5)
    norms = torch.linalg.norm(result.features, dim=1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-6)
```

Update `test_generate_orthogonal_basis_orthogonality`:
```python
def test_generate_orthogonal_basis_orthogonality():
    result = generate_feature_basis(d=10, n=5)
    G = result.features @ result.features.T
    # Off-diagonal should be ~0
    off_diag = G - torch.eye(5)
    assert torch.allclose(off_diag, torch.zeros_like(off_diag), atol=1e-6)
```

Update `test_generate_orthogonal_basis_requires_n_leq_d`:
```python
def test_generate_orthogonal_basis_requires_n_leq_d():
    with pytest.raises(NotImplementedError):
        generate_feature_basis(d=5, n=10)
```

**Step 6: Run all synthetic tests**

Run: `pytest tests/test_synthetic.py -v`

Expected: All PASS

**Step 7: Fix callers of generate_feature_basis**

Update `src/synthetic.py` `generate_representations` call sites if any use the old interface internally. Check and update any notebook or experiment files.

**Step 8: Run full test suite**

Run: `pytest`

Expected: All tests PASS

**Step 9: Commit**

```bash
git add src/synthetic.py tests/test_synthetic.py
git commit -m "refactor: generate_feature_basis returns FeatureBasisResult"
```

---

### Task 4: Implement Gram Matrix Alternating Projection for n > d

**Files:**
- Modify: `src/synthetic.py`
- Test: `tests/test_synthetic.py`

**Step 1: Write failing tests for n > d case**

Add to `tests/test_synthetic.py`:

```python
def test_generate_feature_basis_overcomplete_shape():
    """For n > d, still returns correct shape."""
    from src.synthetic import generate_feature_basis

    result = generate_feature_basis(d=5, n=10)

    assert result.features.shape == (10, 5)


def test_generate_feature_basis_overcomplete_unit_norm():
    """For n > d, features are unit norm."""
    import torch
    from src.synthetic import generate_feature_basis

    result = generate_feature_basis(d=5, n=10)
    norms = torch.linalg.norm(result.features, dim=1)

    assert torch.allclose(norms, torch.ones(10), atol=1e-5)


def test_generate_feature_basis_overcomplete_coherence():
    """For n > d, achieved_epsilon >= welch_bound."""
    from src.synthetic import generate_feature_basis

    result = generate_feature_basis(d=5, n=10)

    assert result.achieved_epsilon >= result.welch_bound - 1e-6
    assert result.welch_bound > 0  # Overcomplete case


def test_generate_feature_basis_overcomplete_coherence_matches():
    """achieved_epsilon matches actual max coherence."""
    import torch
    from src.synthetic import generate_feature_basis

    result = generate_feature_basis(d=5, n=10)
    F = result.features

    G = F @ F.T
    G.fill_diagonal_(0)
    actual_max = G.abs().max().item()

    assert abs(actual_max - result.achieved_epsilon) < 1e-6


def test_generate_feature_basis_overcomplete_near_welch():
    """For reasonable n/d, achieved should be within 2x Welch bound."""
    from src.synthetic import generate_feature_basis

    result = generate_feature_basis(d=32, n=64)

    # Should be reasonably close to Welch bound (within 2x)
    assert result.achieved_epsilon < 2 * result.welch_bound
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_synthetic.py::test_generate_feature_basis_overcomplete_shape -v`

Expected: FAIL with "NotImplementedError"

**Step 3: Write the iterative projection implementation**

Add helper function and update `generate_feature_basis` in `src/synthetic.py`:

```python
import torch.nn.functional as F


def _compute_coherence(features: torch.Tensor) -> float:
    """Compute max off-diagonal absolute coherence."""
    G = features @ features.T
    G.fill_diagonal_(0)
    return G.abs().max().item()


def _iterative_projection(
    d: int,
    n: int,
    max_iters: int = 500,
    shrink_factor: float = 0.9,
    patience: int = 20,
    tolerance: float = 1e-6
) -> FeatureBasisResult:
    """
    Generate n > d features via Gram matrix alternating projection.

    Alternates between:
    1. Shrinking off-diagonal coherence in Gram matrix
    2. Projecting back to valid Gram matrix (PSD, rank d)
    """
    # Initialize with random unit vectors
    features = torch.randn(n, d)
    features = F.normalize(features, dim=1)

    prev_coherence = float('inf')
    stall_count = 0
    converged = False

    for iteration in range(max_iters):
        # Current Gram matrix
        G = features @ features.T

        # Shrink off-diagonal entries toward 0
        off_diag = G - torch.eye(n)
        G_shrunk = torch.eye(n) + shrink_factor * off_diag

        # Project to valid Gram matrix via eigendecomposition
        # Keep top d non-negative eigenvalues
        eigenvalues, eigenvectors = torch.linalg.eigh(G_shrunk)
        eigenvalues = torch.clamp(eigenvalues, min=0)

        # Select top d eigenvalues
        top_d_idx = eigenvalues.argsort(descending=True)[:d]
        selected_vals = eigenvalues[top_d_idx]
        selected_vecs = eigenvectors[:, top_d_idx]

        # Reconstruct features: F = V * sqrt(Lambda)
        features = selected_vecs * selected_vals.sqrt().unsqueeze(0)
        features = F.normalize(features, dim=1)

        # Check convergence
        curr_coherence = _compute_coherence(features)

        if prev_coherence - curr_coherence < tolerance:
            stall_count += 1
            if stall_count >= patience:
                converged = True
                break
        else:
            stall_count = 0

        prev_coherence = curr_coherence

    achieved_eps = _compute_coherence(features)

    return FeatureBasisResult(
        features=features,
        achieved_epsilon=achieved_eps,
        welch_bound=welch_bound(n, d),
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
    For n > d: Uses Gram matrix alternating projection to minimize coherence
    """
    if n <= d:
        # QR decomposition produces orthonormal columns
        random_matrix = torch.randn(d, n)
        q, _ = torch.linalg.qr(random_matrix)
        features = q.T  # (n, d)

        return FeatureBasisResult(
            features=features,
            achieved_epsilon=0.0,
            welch_bound=0.0,
            converged=True
        )
    else:
        return _iterative_projection(d, n)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_synthetic.py::test_generate_feature_basis_overcomplete_shape tests/test_synthetic.py::test_generate_feature_basis_overcomplete_unit_norm tests/test_synthetic.py::test_generate_feature_basis_overcomplete_coherence tests/test_synthetic.py::test_generate_feature_basis_overcomplete_coherence_matches tests/test_synthetic.py::test_generate_feature_basis_overcomplete_near_welch -v`

Expected: All PASS

**Step 5: Run full test suite**

Run: `pytest`

Expected: All PASS

**Step 6: Commit**

```bash
git add src/synthetic.py tests/test_synthetic.py
git commit -m "feat: implement Gram matrix alternating projection for n > d"
```

---

### Task 5: Remove epsilon from SyntheticConfig

**Files:**
- Modify: `src/config.py`
- Modify: `tests/test_config.py`

**Step 1: Update SyntheticConfig**

In `src/config.py`, remove the `epsilon` field from `SyntheticConfig`:

```python
@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    d: int = 64                          # Representation dimension
    n: int = 64                          # Number of features in basis
    # epsilon removed - now derived from generate_feature_basis
    num_representations: int = 1000      # How many to generate

    # ... rest unchanged
```

**Step 2: Update compute_coef_min**

The function uses `config.epsilon`. Update it to accept epsilon as a parameter:

```python
def compute_coef_min(config: SyntheticConfig, epsilon: float = 0.0) -> float:
    """Compute minimum coefficient magnitude: max(factor * k * epsilon, floor)."""
    return max(config.coef_factor * config.k * epsilon, config.coef_min_floor)
```

**Step 3: Update tests**

Update `tests/test_config.py` to reflect new signature:

```python
def test_compute_coef_min_with_epsilon_zero():
    config = SyntheticConfig()
    result = compute_coef_min(config, epsilon=0.0)
    assert result == config.coef_min_floor


def test_compute_coef_min_with_epsilon_nonzero():
    config = SyntheticConfig(coef_factor=10.0, k=3, coef_min_floor=0.1)
    result = compute_coef_min(config, epsilon=0.05)
    # 10 * 3 * 0.05 = 1.5, which is > 0.1
    assert result == 1.5
```

**Step 4: Run tests**

Run: `pytest tests/test_config.py -v`

Expected: PASS

**Step 5: Run full suite to catch any breakages**

Run: `pytest`

Fix any failing tests that were using `config.epsilon`.

**Step 6: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "refactor: remove epsilon from SyntheticConfig (now derived)"
```

---

### Task 6: Update generate_representations Caller

**Files:**
- Modify: `src/synthetic.py`

**Step 1: Check current usage**

The `generate_representations` function calls `compute_coef_min(config)`. Update to pass epsilon:

```python
def generate_representations(
    features: torch.Tensor,
    config: SyntheticConfig,
    epsilon: float = 0.0
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
    # ... rest unchanged
```

**Step 2: Run tests**

Run: `pytest tests/test_synthetic.py -v`

Expected: PASS

**Step 3: Commit**

```bash
git add src/synthetic.py
git commit -m "refactor: generate_representations accepts epsilon parameter"
```

---

### Task 7: Integration Test

**Files:**
- Test: `tests/test_synthetic.py`

**Step 1: Write end-to-end integration test**

Add to `tests/test_synthetic.py`:

```python
def test_full_generation_pipeline():
    """Full pipeline: generate basis -> generate representations."""
    import torch
    from src.synthetic import generate_feature_basis, generate_representations
    from src.config import SyntheticConfig

    # Generate overcomplete basis
    basis = generate_feature_basis(d=10, n=20)

    assert basis.features.shape == (20, 10)
    assert basis.achieved_epsilon > 0  # Overcomplete
    assert basis.welch_bound > 0

    # Generate representations using achieved epsilon
    config = SyntheticConfig(
        d=10,
        n=20,
        num_representations=100,
        k=3,
    )

    representations, coefficients = generate_representations(
        basis.features,
        config,
        epsilon=basis.achieved_epsilon
    )

    assert representations.shape == (100, 10)
    assert coefficients.shape == (100, 20)

    # Verify reconstruction: r = c @ F
    reconstructed = coefficients @ basis.features
    assert torch.allclose(representations, reconstructed, atol=1e-5)
```

**Step 2: Run the test**

Run: `pytest tests/test_synthetic.py::test_full_generation_pipeline -v`

Expected: PASS

**Step 3: Run full suite**

Run: `pytest`

Expected: All PASS

**Step 4: Commit**

```bash
git add tests/test_synthetic.py
git commit -m "test: add full pipeline integration test for overcomplete basis"
```

---

### Task 8: Final Verification

**Step 1: Run full test suite**

Run: `pytest -v`

Expected: All tests PASS

**Step 2: Verify in notebook (manual)**

```python
from src.synthetic import generate_feature_basis, generate_representations
from src.config import SyntheticConfig

# Test overcomplete case
basis = generate_feature_basis(d=64, n=128)
print(f"Welch bound: {basis.welch_bound:.4f}")
print(f"Achieved:    {basis.achieved_epsilon:.4f}")
print(f"Ratio:       {basis.achieved_epsilon / basis.welch_bound:.2f}x")
print(f"Converged:   {basis.converged}")
```

**Step 3: Final commit if any cleanup needed**

```bash
git status
# If clean, done. If changes, commit them.
```

---

## Summary

| Task | Description |
|------|-------------|
| 1 | Add `welch_bound` function |
| 2 | Add `FeatureBasisResult` dataclass |
| 3 | Refactor `generate_feature_basis` for n ≤ d |
| 4 | Implement Gram matrix alternating projection for n > d |
| 5 | Remove `epsilon` from `SyntheticConfig` |
| 6 | Update `generate_representations` signature |
| 7 | Integration test |
| 8 | Final verification |
