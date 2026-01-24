# Epsilon-Orthogonal Feature Basis Generation

## Overview

Generate n unit-norm feature vectors in d dimensions with minimal pairwise coherence, supporting superposition research where n > d.

## Interface

```python
@dataclass
class FeatureBasisResult:
    features: torch.Tensor      # (n, d) unit-norm feature vectors
    achieved_epsilon: float     # actual max |<f_i, f_j>| across all pairs
    welch_bound: float          # theoretical minimum for this n, d
    converged: bool             # True if iteration stabilized

def generate_feature_basis(d: int, n: int) -> FeatureBasisResult:
    """
    Generate n unit-norm feature vectors in d dimensions
    with minimal pairwise coherence.

    For n <= d: Returns orthonormal basis (achieved_epsilon = 0)
    For n > d: Iteratively minimizes max coherence toward Welch bound
    """

def welch_bound(n: int, d: int) -> float:
    """
    Theoretical minimum achievable max coherence for n vectors in d dims.

    Returns 0.0 if n <= d (orthonormal basis possible)
    Returns sqrt((n-d) / (d*(n-1))) if n > d
    """
```

## Algorithm

### Case: n <= d (Orthonormal Basis)

Use QR decomposition to generate orthonormal vectors:

```python
random_matrix = torch.randn(d, n)
q, _ = torch.linalg.qr(random_matrix)
features = q.T  # (n, d) orthonormal rows
```

Result: `achieved_epsilon = 0.0`, `converged = True`

### Case: n > d (Gram Matrix Alternating Projection)

Work with the Gram matrix G = FF^T where G[i,j] = <f_i, f_j>:
- Diagonal entries = 1 (unit norm constraint)
- Off-diagonal entries = coherence (what we minimize)
- Valid G must be positive semidefinite with rank <= d

**Algorithm:**

```python
def _iterative_projection(d: int, n: int, max_iters: int = 500):
    # 1. Initialize random unit vectors
    F = F.normalize(torch.randn(n, d), dim=1)

    prev_coherence = float('inf')
    patience, stall_count, tolerance = 20, 0, 1e-6
    converged = False

    for iteration in range(max_iters):
        G = F @ F.T  # Current Gram matrix

        # 2. Shrink off-diagonal entries (reduce coherence)
        off_diag = G - torch.eye(n)
        shrink_factor = 0.9
        G_shrunk = torch.eye(n) + shrink_factor * off_diag

        # 3. Project back to valid Gram matrix (PSD, rank d)
        eigenvalues, eigenvectors = torch.linalg.eigh(G_shrunk)
        eigenvalues = torch.clamp(eigenvalues, min=0)
        top_d_idx = eigenvalues.argsort(descending=True)[:d]

        # 4. Reconstruct F from truncated eigendecomposition
        F = eigenvectors[:, top_d_idx] * eigenvalues[top_d_idx].sqrt()
        F = F.normalize(F, dim=1)

        # 5. Check convergence
        curr_coherence = (F @ F.T - torch.eye(n)).abs().max().item()
        if prev_coherence - curr_coherence < tolerance:
            stall_count += 1
            if stall_count >= patience:
                converged = True
                break
        else:
            stall_count = 0
        prev_coherence = curr_coherence

    achieved_eps = curr_coherence
    return FeatureBasisResult(F, achieved_eps, welch_bound(n, d), converged)
```

## Integration

### Config Changes

**`src/config.py`** - Remove `epsilon` from `SyntheticConfig`:

```python
@dataclass
class SyntheticConfig:
    d: int = 64
    n: int = 64
    # epsilon removed - now derived from generation
    num_representations: int = 1000
    # ... rest unchanged
```

### Usage Flow

```python
# Generate features (epsilon is discovered, not specified)
basis = generate_feature_basis(config.d, config.n)
print(f"Achieved eps={basis.achieved_epsilon:.4f} (Welch={basis.welch_bound:.4f})")

# Pass achieved epsilon to extraction config
ext_config = ExtractionConfig(
    epsilon=basis.achieved_epsilon,  # Now calibrated correctly
    # ...
)

# Generate representations as before
representations, coefficients = generate_representations(basis.features, config)
```

Key insight: epsilon flows **from** generation **to** extraction, not the other way around.

## Validation

```python
def validate_basis(result: FeatureBasisResult, d: int, n: int):
    F = result.features

    # Shape check
    assert F.shape == (n, d)

    # Unit norm check
    norms = torch.linalg.norm(F, dim=1)
    assert torch.allclose(norms, torch.ones(n), atol=1e-5)

    # Coherence matches reported epsilon
    G = F @ F.T
    G.fill_diagonal_(0)
    actual_max = G.abs().max().item()
    assert abs(actual_max - result.achieved_epsilon) < 1e-6

    # Achieved >= Welch bound (can't beat physics)
    assert result.achieved_epsilon >= result.welch_bound - 1e-6
```

## Expected Output

```
Generated 100 features in 64 dims
  Welch bound:      0.0756
  Achieved epsilon: 0.0812 (1.07x Welch)
  Converged:        True (iter 234/500)
```

## References

- [Grassmannian Frame via Alternating Projections](https://openreview.net/pdf?id=vng6moOJ9L)
- [Algorithms for Construction of Incoherent Frames](https://arxiv.org/abs/1801.09678)
