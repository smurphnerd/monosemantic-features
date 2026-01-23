# Target Selection via Neighborhood Minimality

## Overview

Implement the neighborhood minimality criterion from `writeup/sec/3_method.tex` (Section 3.1) to select monosemantic representations as feature extraction targets.

**Problem:** Polysemantic representations (containing multiple features) yield neighbor sets that are unions of multiple feature neighborhoods. Projecting these onto the nullspace produces multi-rank matrices, and SVD returns rotated vectors rather than true feature directions.

**Solution:** Select targets whose neighbor set cardinality is a local minimum relative to their neighbors. These are likely monosemantic (single-feature) and yield rank-1 projections for clean feature extraction.

**Criterion:** A representation r is a valid target if:
```
|X_pos| <= |X_pos,i| for all neighbors r_i in X_pos
```

## New Functions

### `build_neighbor_matrix(X, tau) -> np.ndarray`

Location: `src/extraction.py`

**Input:**
- `X`: representations matrix (n x d), row-major
- `tau`: cosine similarity threshold

**Output:**
- Boolean adjacency matrix (n x n) where `matrix[i,j] = |cossim(X[i], X[j])| >= tau`
- Diagonal is True (self-neighbors)
- Matrix is symmetric

**Implementation:**
```python
def build_neighbor_matrix(X: np.ndarray, tau: float) -> np.ndarray:
    # Normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / norms

    # Pairwise cosine similarities
    cossim = X_norm @ X_norm.T

    # Threshold with absolute value
    neighbor_matrix = np.abs(cossim) >= tau

    return neighbor_matrix
```

### `find_monosemantic_targets(neighbor_matrix) -> np.ndarray`

Location: `src/extraction.py`

**Input:**
- `neighbor_matrix`: boolean adjacency matrix (n x n) from `build_neighbor_matrix`

**Output:**
- Array of indices for representations passing the minimality criterion
- Deduplicated: only one representative per unique neighbor set

**Implementation:**
```python
def find_monosemantic_targets(neighbor_matrix: np.ndarray) -> np.ndarray:
    n = neighbor_matrix.shape[0]
    neighbor_counts = neighbor_matrix.sum(axis=1)

    # Group by identical neighbor sets (rows)
    seen_rows = {}
    representatives = []

    for i in range(n):
        row_key = neighbor_matrix[i].tobytes()
        if row_key not in seen_rows:
            seen_rows[row_key] = i
            representatives.append(i)

    # Filter by minimality criterion
    targets = []
    for i in representatives:
        neighbors = neighbor_matrix[i]
        min_neighbor_count = neighbor_counts[neighbors].min()
        if neighbor_counts[i] <= min_neighbor_count:
            targets.append(i)

    return np.array(targets, dtype=int)
```

## Modified Functions

### `extract_feature` - Updated Signature

**Old:**
```python
def extract_feature(X, target_idx, tau, epsilon) -> np.ndarray | None
```

**New:**
```python
def extract_feature(
    X: np.ndarray,
    neighbor_mask: np.ndarray,  # boolean array (n,)
    config: ExtractionConfig,
) -> np.ndarray | None
```

**Changes:**
- Accepts precomputed boolean mask instead of target index and tau
- Derives `X_pos = X[neighbor_mask]` and `X_neg = X[~neighbor_mask]`
- Uses `config.epsilon` for nullspace threshold
- Internal nullspace/projection logic unchanged

### `extract_all_features` - New Parameter

**Updated signature:**
```python
def extract_all_features(
    X: np.ndarray,
    config: ExtractionConfig,
    use_minimality_filter: bool = True,  # NEW
) -> np.ndarray
```

**Updated logic:**
```python
def extract_all_features(
    X: np.ndarray,
    config: ExtractionConfig,
    use_minimality_filter: bool = True,
) -> np.ndarray:
    # Build neighbor matrix upfront
    neighbor_matrix = build_neighbor_matrix(X, config.tau)

    if use_minimality_filter:
        # New path: minimality-based selection
        target_indices = find_monosemantic_targets(neighbor_matrix)
    else:
        # Old path: all unique neighbor sets
        target_indices = _get_all_cluster_representatives(neighbor_matrix)

    features = []
    for idx in target_indices:
        neighbor_mask = neighbor_matrix[idx]
        feature = extract_feature(X, neighbor_mask, config)
        if feature is not None:
            features.append(feature)

    # Deduplicate extracted features
    return deduplicate_features(np.array(features))
```

## Data Flow

```
X (representations)
    |
    v
build_neighbor_matrix(X, tau)
    |
    v
neighbor_matrix (n x n bool)
    |
    +-- use_minimality_filter=True --> find_monosemantic_targets()
    |                                       |
    +-- use_minimality_filter=False --> _get_all_cluster_representatives()
                                            |
                                            v
                                    target_indices (deduplicated)
                                            |
                                            v
                            for each target: extract_feature(X, neighbor_mask, config)
                                            |
                                            v
                                    deduplicate_features()
                                            |
                                            v
                                    extracted features
```

## Testing Strategy

### Unit Tests

**`test_build_neighbor_matrix`**
- Verify diagonal is True
- Verify symmetry (matrix[i,j] == matrix[j,i])
- Verify threshold behavior with known pairs above/below tau

**`test_find_monosemantic_targets`**
- Hand-crafted small graph where minimality is clear
- Verify deduplication: identical rows yield single representative
- Edge case: all representations are monosemantic (sparse features)
- Edge case: no representations pass filter (all polysemantic)

### Integration Tests

**`test_minimality_filter_finds_monosemantic`**
- Generate mixed data: ~50% k=1 (monosemantic), ~50% k=2+ (polysemantic)
- Verify `find_monosemantic_targets` returns indices corresponding to k=1 representations
- Verify polysemantic representations are excluded

**`test_minimality_improves_alignment`**
- With `use_minimality_filter=True`: extract from monosemantic targets, expect high alignment
- With `use_minimality_filter=False`: extract from all clusters, expect lower alignment due to rotated features
- Assert filter yields better mean alignment

**`test_all_polysemantic_returns_empty`**
- Generate data with all k >= 2
- Verify `find_monosemantic_targets` returns empty array
- Verify `extract_all_features` with filter returns empty (no bad extractions)

**`test_rank1_projection`**
- For targets passing minimality, verify projected neighbor matrix is approximately rank-1
- For targets failing minimality, verify higher rank (multiple significant singular values)

## Files to Modify

1. `src/extraction.py` - Add new functions, update signatures
2. `tests/test_extraction.py` - Add unit tests for new functions
3. `tests/test_integration.py` - Add integration tests with mixed sparsity data
4. `experiments/run_experiment.py` - Update to use new `extract_all_features` signature

## Migration Notes

- Default `use_minimality_filter=True` enables new behavior
- Set `use_minimality_filter=False` to restore old behavior for comparison
- Once validated, the flag can be removed and minimality becomes the only path
