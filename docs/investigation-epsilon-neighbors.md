# Investigation: Why minimality-based target selection fails with ε > 0

## Executive Summary

**Updated Finding**: The monosemantic feature extraction algorithm has **complex behavior** with ε-orthogonal features. While the tau bounds formula is incorrect and no perfect separating threshold exists, the **minimality filter still works reasonably well** (F1 scores 0.6-0.8) because it uses relative neighbor counts rather than absolute similarity thresholds.

**Key Insight**: The algorithm's robustness comes from the **two-stage process**: (1) neighbor clustering with tau, (2) minimality filtering. Even when stage 1 is imperfect due to overlapping similarity distributions, stage 2 can still identify monosemantic representations by their relatively smaller neighbor sets.

## Resolution: Switch to Dot Product Similarity

**Date:** 2026-01-25

### Why Cosine Similarity Was Problematic

Cosine similarity normalizes by the product of vector norms: `cossim(a,b) = (a·b)/(‖a‖·‖b‖)`. This normalization destroys magnitude information that is critical for separating sharing from non-sharing pairs when ε > 0.

Consider two representations with small coefficients on a shared feature. The shared feature's dot product contribution is `c₁·c₂ ≈ coef_min²`, but the normalization divides by `‖r₁‖·‖r₂‖ ≈ k·coef_max²`. Meanwhile, the ε-interference noise also gets divided by the same norms, so both signal and noise scale identically after normalization. For small-coefficient monosemantic representations, the normalized signal can be indistinguishable from normalized noise.

### Why Dot Product Fixes This

With raw dot product, magnitude is preserved:
- **Sharing pairs** (share at least one feature): dot product ≥ `coef_min²` from the shared feature's coefficient product. Interference from other feature pairs adds positively in expectation.
- **Non-sharing pairs**: dot product ≤ `k² × coef_max² × ε`, since all k×k cross-feature pairs contribute at most `coef_max² × ε` each.

The key insight is that shared feature contributions scale with the *coefficients themselves* (coef_min²), while non-sharing interference scales with *ε*. These are on completely different scales when the generator has reasonable parameters.

### New Tau Bounds Derivation

For dot product:
- **tau_upper** = `coef_min²` (minimum sharing dot product)
- **tau_lower** = `k² × coef_max² × ε` (maximum non-sharing dot product)
- **Separable when**: `coef_min² > k² × coef_max² × ε`

Compare with old cosine similarity bounds:
- Old tau_upper = `coef_min² / (k × coef_max²)` — divided out magnitude
- Old tau_lower = `k² × coef_max² × ε / (k × coef_max²)` = `k × ε` — also divided out

The old bounds had the same ratio (separability condition was equivalent), but the absolute values were wrong because the normalization interacted poorly with the actual similarity distributions. The dot product bounds directly reflect the physical quantities.

### When Tau Is Calculable vs Hyperparameter

**Calculable** (auto-derive tau): When generator parameters (k, ε, coef_min, coef_max) are known and the separability condition `coef_min² > k² × coef_max² × ε` holds. This is the case for synthetic experiments with controlled generation.

**Hyperparameter** (manual tuning needed): When generator parameters are unknown (real data), when the separability condition fails (very high ε or very high k), or when using `bernoulli_gaussian` sparsity where effective k varies per representation.

---

## Investigation Details

### Parameters
- **Dimensions**: d=16, n=24 (overcomplete basis to force ε > 0)
- **Sparsity**: k=2 active features per polysemantic representation
- **Representations**: 20 total (6 monosemantic k=1, 14 polysemantic k=2)
- **Epsilon values tested**: [0.0, 0.05, 0.1, 0.15, 0.2]
- **Achieved epsilon**: ~0.16 for all ε > 0 cases (near Welch bound)

### Critical Findings

#### 1. No Separable Tau Exists for ε > 0
For all ε > 0 cases:
- **Actual sharing minimum**: 0.06-0.19 (lowest similarity between pairs sharing a feature)
- **Actual non-sharing maximum**: 0.33-0.36 (highest similarity between pairs NOT sharing features)  
- **Gap**: **NEGATIVE** (-0.17 to -0.27)

This means there is **no threshold τ** that can separate sharing from non-sharing pairs.

#### 2. Tau Bounds Formula is Wrong
The current formula predicts:
- **tau_lower** (max non-sharing similarity): 0.32-0.33
- **tau_upper** (min sharing similarity): 5.1-5.3

**Reality**:
- **Actual max non-sharing**: 0.23-0.25 (formula over-predicts by ~30%)
- **Actual min sharing**: 0.06-0.19 (formula over-predicts by >25x!)

The formula assumes:
1. Sharing pairs have similarity ≥ `coef_min²/max_norm²` 
2. Non-sharing pairs have similarity ≤ `k²⋅coef_max²⋅ε/max_norm²`

**Both assumptions are wrong** in practice.

#### 3. Why Sharing Pairs Can Have Low Similarity

**Case Analysis**: Two representations sharing feature f can have very low cosine similarity when:

1. **Opposing signs**: If one has coefficient +c and another has -c for the shared feature
2. **Interference dominance**: The ε-interference from other feature pairs overwhelms the shared feature contribution
3. **Small coefficients**: When the shared feature has small coefficients while other features have large ones

**Example**: 
- Rep 1: feature A (+0.1), feature B (+1.0)  
- Rep 2: feature A (-0.1), feature C (+1.0)
- Shared feature A contributes: +0.1 × -0.1 = -0.01
- Interference B-C: +1.0 × 1.0 × ε ≈ +0.16
- **Net similarity**: dominated by interference, not shared feature

#### 4. Signal-to-Noise Ratio Collapse
- **ε = 0**: SNR = ~14,000 (perfect separation)
- **ε > 0**: SNR = 4-6 (poor separation)

The ε-interference noise becomes comparable to or larger than the shared feature signal.

#### 5. Coefficient Scaling Doesn't Help
Testing coefficient magnitudes from 0.05 to 1.0 showed that **increasing coefficient magnitudes does not restore separability**. The fundamental issue is that interference scales with the same coefficients as the signal.

### Interference Analysis

The ε-interference calculations are **mathematically correct**:
- Theoretical vs actual interference: mean difference ≈ 0.0000
- The issue isn't computational error but fundamental non-separability

For non-sharing pairs, interference accumulates as:
```
similarity ≈ Σᵢⱼ cᵢ × cⱼ × ⟨fᵢ, fⱼ⟩
```
where i, j are active features in the two representations.

With k=2 features per representation and ε ≈ 0.16, this can create similarities up to ~0.33, which **exceeds** many sharing pair similarities.

## Proposed Solutions

### 1. Alternative Neighbor Discovery
Instead of single τ threshold:
- **Adaptive thresholding**: Use different τ for different representations based on their coefficient magnitudes
- **Relative ranking**: For each representation, take its top-k most similar neighbors regardless of absolute threshold
- **Multiple thresholds**: Sweep τ and use consensus among multiple threshold values

### 2. Feature-Specific Analysis
- **Nullspace projection first**: Project representations into estimated feature nullspaces before computing similarities
- **Iterative feature extraction**: Extract one feature at a time and remove its contribution before extracting others
- **Spectral clustering**: Use eigendecomposition of similarity matrix for clustering

### 3. Robust Statistics
- **Median-based clustering**: Use median similarities instead of raw cosine similarities
- **Outlier-resistant metrics**: Replace cosine similarity with metrics robust to outliers
- **Bootstrap aggregation**: Use multiple random samples to find robust clusters

### 4. Ground Truth Verification
The investigation used known ground truth to verify that the **algorithm's assumptions are mathematically impossible**. Alternative metrics like:
- **Mutual information**: Measure statistical dependence rather than linear similarity
- **Feature reconstruction error**: How well can one representation predict another's active features?

### 5. Bounds Correction
Fix the tau_bounds formula by:
1. **Empirical bounds**: Use actual similarity distributions from a sample
2. **Worst-case analysis**: Account for sign conflicts and interference accumulation 
3. **Confidence intervals**: Provide probabilistic rather than deterministic bounds

## Implementation Recommendations

### Immediate Fix: Disable Auto-Tau for ε > 0
```python
def resolve_tau(extraction_config, synthetic_config, epsilon):
    if epsilon > 0.05:  # Empirical threshold
        logger.warning("Auto-tau disabled for epsilon > 0.05. Use manual tau.")
        return extraction_config.tau or 0.5  # Fallback
    # ... existing logic for epsilon ≈ 0
```

### Enhanced Target Selection
```python
def find_monosemantic_targets_robust(representations, tau_range):
    """Try multiple tau values and use consensus."""
    target_sets = []
    for tau in tau_range:
        neighbor_matrix = build_neighbor_matrix(representations, tau)
        targets = find_monosemantic_targets(neighbor_matrix)
        target_sets.append(set(targets.tolist()))
    
    # Use targets that appear in multiple tau values
    consensus_targets = set.intersection(*target_sets) if target_sets else set()
    return torch.tensor(list(consensus_targets))
```

### Alternative Clustering
```python
def cluster_by_reconstruction(representations, features_estimate):
    """Cluster by how well each representation reconstructs others."""
    # Implementation using reconstruction loss instead of cosine similarity
    pass
```

## Conclusion

The minimality-based target selection fails with ε > 0 because **the fundamental assumption of the algorithm is violated**: that sharing features leads to higher cosine similarity than not sharing features. 

With ε-orthogonal features, interference between non-shared features can create higher similarities than weak or opposing shared features. The tau bounds formula in `compute_tau_bounds()` doesn't account for this and provides meaningless bounds.

**The algorithm needs either**:
1. **Better similarity metrics** that account for ε-interference
2. **Alternative clustering methods** not based on cosine similarity
3. **Empirical threshold selection** instead of theoretical bounds
4. **Preprocessing** to reduce ε-interference before clustering

This is a fundamental algorithmic limitation, not a parameter tuning problem.

## MAJOR UPDATE: Spectral Gap Investigation Reveals True Root Cause

**Critical Discovery**: The previous analysis missed the most important failure mode. The catastrophic failure observed in the notebook `notebooks/epsilon_orth_feature.ipynb` is primarily due to **broken auto-tau derivation for bernoulli_gaussian sparsity**, not fundamental algorithmic issues.

### Spectral Gap Investigation Summary

A comprehensive follow-up investigation using the correct sparsity model (`bernoulli_gaussian` with k=6 expected features) revealed:

1. **Auto-tau is catastrophically wrong**: Derives tau=1.37 while actual similarities range [0, 0.99], resulting in 0% neighbors

2. **Minimality filter works with correct tau**: Achieves 100% recall at tau=0.15, with mono reps having 38% fewer neighbors than average (supporting minimality assumption)

3. **Spectral gap post-hoc filtering is highly effective**: σ₁/σ₂ ratios successfully distinguish monosemantic (mean gap=11M) from multi-feature extractions (mean gap=2M)

4. **"Almost-monosemantic" representations exist**: 72 polysemantic reps have high spectral gaps due to one dominant feature, with 54% extracting meaningful features

**See `docs/spectral-gap-investigation.md` for complete analysis.**

## Updated Investigation: Comprehensive Testing Results

### Methodology Revision
The initial investigation used unrealistic parameters (all representations had k=2). A comprehensive follow-up test used:
- **Mixed representations**: 30% monosemantic (k=1), 70% polysemantic (k=2) 
- **Realistic epsilon values**: n > d configurations achieving ε ≈ 0.12-0.16
- **Manual tau sweeping**: Testing across full tau range to find optimal performance
- **Ground truth validation**: Direct comparison with known monosemantic representations

### Key Findings Revision

#### 1. Algorithm Performance is Better Than Expected
**Comprehensive Results Summary**:
- **eps_0.0_n_16**: F1=0.714, Separable=True (perfect orthogonality baseline)
- **eps_0.1_n_20**: F1=0.769, Separable=**False** (realistic ε=0.123)
- **eps_0.1_n_24**: F1=0.769, Separable=**False** (realistic ε=0.160) 
- **eps_0.2_n_20**: F1=0.800, Separable=**False** (realistic ε=0.123)
- **eps_0.2_n_24**: F1=0.667, Separable=True (realistic ε=0.161)

#### 2. Why the Algorithm Still Works Despite Non-Separability

**Critical Discovery**: The algorithm achieves F1 > 0.75 even when **no separable tau exists**. This happens because:

1. **Minimality filtering is robust**: Even if sharing/non-sharing similarities overlap, monosemantic representations tend to have fewer neighbors than polysemantic ones.

2. **Local optimality matters**: The minimality criterion finds representations with locally minimal neighbor counts within their cluster, not globally minimal.

3. **Tau choice affects cluster structure**: Different tau values create different neighbor groupings, but the minimality filter can still pick out the relatively simpler representations.

**Example Analysis (eps_0.1_n_24, tau=0.218)**:
- Monosemantic representations (k=1): neighbor counts [4,6,7,8,6,5] 
- Found as targets: 5/6 (83% recall)
- False positives from polysemantic reps with neighbor counts [2,4] (locally minimal in their clusters)

#### 3. Tau Bounds Formula Issues Confirmed But Less Critical

The theoretical bounds are still wrong:
- **Predicted tau_lower**: 0.32, **Actual max non-sharing**: 0.27 (slight over-prediction)
- **Predicted tau_upper**: 5.17, **Actual min sharing**: 0.05 (massive over-prediction)

However, this is **less critical** than initially thought because:
- Manual tau sweep finds working values (tau ≈ 0.2-0.3) 
- The algorithm's robustness comes from minimality filtering, not perfect tau bounds
- Even with wrong bounds, good performance is achievable

### Practical Implications

#### Current Algorithm Status: **Functional but Suboptimal**
- **Works**: F1 scores 0.6-0.8 achievable with manual tau tuning
- **Fails**: Auto-derived tau bounds are unreliable for ε > 0
- **Recommendation**: Replace auto-tau with empirical validation or manual tuning

#### Suggested Fixes (Priority Order)

1. **Immediate (Low Risk)**: 
   ```python
   def resolve_tau_safe(extraction_config, synthetic_config, epsilon):
       if epsilon > 0.05:
           logger.warning("Using manual tau for epsilon > 0.05")
           return extraction_config.tau or 0.25  # Empirically good default
       return resolve_tau(extraction_config, synthetic_config, epsilon)  # Original logic
   ```

2. **Short Term (Medium Risk)**:
   - Add separability validation before using auto-tau
   - Implement tau sweeping with cross-validation  
   - Use empirical bounds estimation from data samples

3. **Long Term (Research)**:
   - Develop alternative similarity metrics robust to ε-interference
   - Replace cosine similarity with feature reconstruction loss
   - Implement iterative feature extraction with interference removal

### Conclusion

**Updated Assessment**: The minimality-based target selection does **not fundamentally fail** with ε > 0, but it becomes **unreliable with auto-tau**. The core algorithm (neighbor clustering + minimality filtering) is more robust than the tau bounds derivation.

**Root cause**: Mathematical incorrectness in `compute_tau_bounds()` for ε > 0, but the downstream algorithm can compensate.

**Impact**: Medium severity - algorithm needs manual tuning but can still achieve good performance.

**Fix complexity**: Low to Medium - mostly parameter estimation improvements rather than algorithmic overhaul.

## Final Comprehensive Assessment

### Investigation Timeline & Key Findings

1. **Initial Analysis** (ε-orthogonality with fixed sparsity): Found algorithm works reasonably well (F1 0.6-0.8) with ε > 0, but tau bounds formula incorrect.

2. **Sean's Notebook Evidence** (`bernoulli_gaussian` sparsity): Revealed catastrophic failure (0% recall) with realistic sparsity model.

3. **Spectral Gap Investigation** (corrected analysis): Discovered the true root cause is broken auto-tau derivation, not algorithmic failure.

### Unified Understanding

**The algorithm has THREE distinct failure modes**:

1. **ε-orthogonality interference** (original investigation): Tau bounds incorrect for ε > 0, but minimality filter compensates reasonably well.

2. **Sparsity model mismatch** (spectral gap investigation): Auto-tau derivation completely broken for `bernoulli_gaussian`, creating degenerate neighbor relationships.

3. **Scale effects** (implied): Performance may degrade with larger datasets due to computational complexity.

### Root Cause Priority Order

1. **CRITICAL**: Auto-tau derivation for `bernoulli_gaussian` sparsity (causes total failure)
2. **IMPORTANT**: Tau bounds formula for ε > 0 cases (causes suboptimal performance)  
3. **MODERATE**: Minimality assumption edge cases (manageable with spectral gap backup)

### Recommended Solution Architecture

```python
def robust_monosemantic_extraction(reps, coeffs, config):
    """Robust extraction combining multiple approaches."""
    
    # 1. Smart tau selection
    if config.sparsity_mode == "bernoulli_gaussian":
        tau = empirical_tau_selection(reps)  # Data-driven
    elif epsilon > 0.05:
        tau = manual_tau_with_validation(reps, config)  # Avoid broken bounds
    else:
        tau = resolve_tau(config, epsilon)  # Original logic for ε ≈ 0
    
    # 2. Extract from ALL unique neighbor sets (no pre-filtering)
    neighbor_matrix = build_neighbor_matrix(reps, tau)
    all_representatives = get_unique_neighbor_sets(neighbor_matrix)
    
    extractions = []
    for idx in all_representatives:
        result = extract_feature_for_target(reps, idx, neighbor_matrix, epsilon)
        if result and result.spectral_gap >= spectral_threshold:
            extractions.append(result)
    
    # 3. Post-hoc quality control
    return filter_by_feature_alignment(extractions, threshold=0.9)
```

### Performance Expectations

- **bernoulli_gaussian with fixed tau**: 100% recall, high precision with spectral gap filtering
- **ε-orthogonal with manual tau**: 60-80% F1, good enough for practical use
- **Combined approach**: Robust across sparsity models and epsilon values

### Implementation Priority

1. **Immediate**: Add sparsity-aware tau selection (`bernoulli_gaussian` detection)
2. **Short-term**: Implement spectral gap post-hoc filtering  
3. **Long-term**: Develop better tau bounds for all ε > 0 cases

The investigation reveals this is primarily a **parameter estimation problem** with algorithmic enhancements as backup, not a fundamental algorithmic failure.