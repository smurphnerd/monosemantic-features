#!/usr/bin/env python3
"""
Validate dot product vs cosine similarity for neighbor graph construction.

Tests three regimes:
1. ε=0 (orthogonal features) — both metrics work, dot product bounds exact
2. ε>0, separable params — dot product bounds predict separability correctly
3. ε>0, non-separable params — bounds correctly predict failure
"""

import torch
import sys
sys.path.insert(0, ".")

from src.synthetic import generate_feature_basis, generate_representations
from src.config import SyntheticConfig, compute_coef_min
from src.extraction import compute_tau_bounds


def analyze_separation(representations, coefficients, feature_basis, label):
    """Compute dot products and cosine sims, analyze separation."""
    n = representations.shape[0]
    num_features = feature_basis.shape[0]

    # Active features per representation
    active_features = []
    for i in range(n):
        active = set(torch.where(coefficients[i].abs() > 1e-8)[0].tolist())
        active_features.append(active)

    # Pairwise metrics
    dots_all = representations @ representations.T
    norms = torch.norm(representations, dim=1)
    cossim_all = dots_all / (norms.unsqueeze(1) * norms.unsqueeze(0) + 1e-8)

    sharing_dots, nonsharing_dots = [], []
    sharing_cos, nonsharing_cos = [], []

    for i in range(n):
        for j in range(i + 1, n):
            d = abs(float(dots_all[i, j]))
            c = abs(float(cossim_all[i, j]))
            if active_features[i] & active_features[j]:
                sharing_dots.append(d)
                sharing_cos.append(c)
            else:
                nonsharing_dots.append(d)
                nonsharing_cos.append(c)

    print(f"\n--- {label} ---")
    print(f"  Pairs: {len(sharing_dots)} sharing, {len(nonsharing_dots)} non-sharing")

    for name, s, ns in [("Dot product", sharing_dots, nonsharing_dots),
                         ("Cosine sim", sharing_cos, nonsharing_cos)]:
        if s and ns:
            gap = min(s) - max(ns)
            sep = "✓ SEPARABLE" if gap > 0 else "✗ overlap"
            print(f"  {name:12s}: sharing [{min(s):.4f}, {max(s):.4f}], "
                  f"non-sharing [{min(ns):.4f}, {max(ns):.4f}], gap={gap:+.4f} {sep}")
        elif s:
            print(f"  {name:12s}: sharing [{min(s):.4f}, {max(s):.4f}], no non-sharing pairs")

    return sharing_dots, nonsharing_dots


def main():
    torch.manual_seed(42)

    # ========== TEST 1: ε=0, orthogonal features ==========
    print("=" * 65)
    print("TEST 1: Orthogonal features (ε=0)")
    print("=" * 65)

    d = 16
    basis1 = generate_feature_basis(d, d)  # n=d → orthogonal
    eps1 = basis1.achieved_epsilon
    print(f"d={d}, n={d}, ε={eps1:.6f}")

    config1 = SyntheticConfig(
        d=d, n=d, k=2, num_representations=20,
        coef_min_floor=0.3, coef_max=1.0, coef_factor=0.0,  # factor=0 → use floor
        positive_only=True
    )
    reps1, coeffs1 = generate_representations(basis1.features, config1, epsilon=eps1)

    actual_cmin1 = compute_coef_min(config1, eps1)
    tau_upper1, tau_lower1 = compute_tau_bounds(config1.k, eps1, actual_cmin1, config1.coef_max)
    print(f"coef_min={actual_cmin1:.4f}, coef_max={config1.coef_max}")
    print(f"Tau bounds: upper={tau_upper1:.4f}, lower={tau_lower1:.4f}, separable={tau_lower1 < tau_upper1}")

    s1, ns1 = analyze_separation(reps1, coeffs1, basis1.features, "ε=0")

    if s1 and ns1:
        print(f"\n  Bounds check:")
        print(f"    tau_upper ({tau_upper1:.4f}) ≤ min sharing dot ({min(s1):.4f})? {tau_upper1 <= min(s1) + 1e-6}")
        print(f"    tau_lower ({tau_lower1:.4f}) ≥ max nonsharing dot ({max(ns1):.4f})? {tau_lower1 >= max(ns1) - 1e-6}")

    # ========== TEST 2: ε>0, controlled coefficients ==========
    print(f"\n{'=' * 65}")
    print("TEST 2: ε-orthogonal features, tight coefficient range")
    print("=" * 65)

    n_feat = 20
    basis2 = generate_feature_basis(d, n_feat)
    eps2 = basis2.achieved_epsilon
    print(f"d={d}, n={n_feat}, ε={eps2:.4f}")

    # Use coef_factor=0 so coef_min = floor, giving predictable bounds
    config2 = SyntheticConfig(
        d=d, n=n_feat, k=2, num_representations=20,
        coef_min_floor=0.5, coef_max=1.0, coef_factor=0.0,
        positive_only=True
    )
    reps2, coeffs2 = generate_representations(basis2.features, config2, epsilon=eps2)

    actual_cmin2 = compute_coef_min(config2, eps2)
    tau_upper2, tau_lower2 = compute_tau_bounds(config2.k, eps2, actual_cmin2, config2.coef_max)
    print(f"coef_min={actual_cmin2:.4f}, coef_max={config2.coef_max}")
    print(f"Tau bounds: upper={tau_upper2:.4f}, lower={tau_lower2:.4f}, separable={tau_lower2 < tau_upper2}")

    s2, ns2 = analyze_separation(reps2, coeffs2, basis2.features, "ε>0, coef_factor=0")

    if s2 and ns2:
        print(f"\n  Bounds check:")
        print(f"    tau_upper ({tau_upper2:.4f}) ≤ min sharing dot ({min(s2):.4f})? {tau_upper2 <= min(s2) + 1e-6}")
        print(f"    tau_lower ({tau_lower2:.4f}) ≥ max nonsharing dot ({max(ns2):.4f})? {tau_lower2 >= max(ns2) - 1e-6}")

    # ========== TEST 3: ε>0, large coef_factor (auto-scaling) ==========
    print(f"\n{'=' * 65}")
    print("TEST 3: ε-orthogonal, auto-scaled coefficients (coef_factor=10)")
    print("=" * 65)

    config3 = SyntheticConfig(
        d=d, n=n_feat, k=2, num_representations=20,
        coef_min_floor=0.3, coef_max=1.0, coef_factor=10.0,
        positive_only=True
    )
    reps3, coeffs3 = generate_representations(basis2.features, config3, epsilon=eps2)

    actual_cmin3 = compute_coef_min(config3, eps2)
    actual_cmax3 = max(config3.coef_max, actual_cmin3)
    tau_upper3, tau_lower3 = compute_tau_bounds(config3.k, eps2, actual_cmin3, actual_cmax3)
    print(f"coef_min={actual_cmin3:.4f}, coef_max(effective)={actual_cmax3:.4f}")
    print(f"Tau bounds: upper={tau_upper3:.4f}, lower={tau_lower3:.4f}, separable={tau_lower3 < tau_upper3}")
    print(f"Note: coef_factor pushes coef_min >> coef_max, narrowing coefficient range")

    s3, ns3 = analyze_separation(reps3, coeffs3, basis2.features, "ε>0, coef_factor=10")

    if s3 and ns3:
        print(f"\n  Bounds check:")
        print(f"    tau_upper ({tau_upper3:.4f}) ≤ min sharing dot ({min(s3):.4f})? {tau_upper3 <= min(s3) + 1e-6}")
        print(f"    tau_lower ({tau_lower3:.4f}) ≥ max nonsharing dot ({max(ns3):.4f})? {tau_lower3 >= max(ns3) - 1e-6}")

    # ========== SUMMARY ==========
    print(f"\n{'=' * 65}")
    print("SUMMARY")
    print("=" * 65)
    print("With dot product, tau bounds are:")
    print("  tau_upper = coef_min²")
    print("  tau_lower = k² × coef_max² × ε")
    print("Separable when coef_min² > k² × coef_max² × ε")
    print()
    print("Key advantage over cosine similarity:")
    print("  - Dot product preserves magnitude → shared signal scales with coef²")
    print("  - Cosine sim normalizes magnitude → signal/noise ratio collapses")
    print("  - Bounds directly reflect generator params without norm estimation")


if __name__ == "__main__":
    main()
