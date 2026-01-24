"""
Investigate cluster contamination with k>1.

Run with: python -m experiments.investigate_clusters
"""
import torch
from src.config import SyntheticConfig, ExtractionConfig
from src.synthetic import generate_feature_basis, generate_representations
from src.extraction import (
    cluster_by_neighbors,
    resolve_tau,
    compute_nullspace,
    extract_feature,
)


def get_active_features(coef_row: torch.Tensor) -> set[int]:
    return set(torch.where(coef_row != 0)[0].tolist())


def main():
    torch.manual_seed(42)

    syn_config = SyntheticConfig(
        d=16,
        n=16,
        epsilon=0.0,
        num_representations=1000,
        sparsity_mode="fixed",
        k=3,
        coef_min_floor=0.5,
        positive_only=True,  # No sign cancellation
    )
    ext_config = ExtractionConfig(
        tau=0.5,           # High tau: only very similar neighbors
        neg_tau=0.05,      # Only non-neighbors with |cos_sim| <= 0.05 to neighbor mean
        epsilon=0.0,
    )

    print("=== Configuration ===")
    print(f"d={syn_config.d}, n={syn_config.n}, k={syn_config.k}")
    print(f"num_representations={syn_config.num_representations}")
    print(f"coef_min_floor={syn_config.coef_min_floor}")
    print(f"tau={ext_config.tau}, neg_tau={ext_config.neg_tau}")
    print()

    features = generate_feature_basis(syn_config.d, syn_config.n, syn_config.epsilon)
    representations, coefficients = generate_representations(features, syn_config)

    repr_features = [
        get_active_features(coefficients[i])
        for i in range(syn_config.num_representations)
    ]

    tau = resolve_tau(ext_config, syn_config)
    print(f"=== Auto-derived tau: {tau:.4f} ===")
    print()

    # Analyze cosine similarities between representations sharing each feature
    print("=== Cosine Similarity Within Feature Groups ===")
    norms = torch.norm(representations, dim=1)
    for feature_id in range(min(5, syn_config.n)):  # First 5 features
        # Find all representations that have this feature
        has_feature = [
            i
            for i in range(syn_config.num_representations)
            if feature_id in repr_features[i]
        ]
        if len(has_feature) < 2:
            continue

        # Compute pairwise cosine similarities
        group_reprs = representations[has_feature]
        group_norms = norms[has_feature]
        cos_sims = (group_reprs @ group_reprs.T) / (
            group_norms[:, None] * group_norms[None, :] + 1e-8
        )

        # Get off-diagonal similarities
        mask = ~torch.eye(len(has_feature), dtype=torch.bool)
        pairwise_sims = cos_sims[mask]

        abs_sims = pairwise_sims.abs()
        print(f"Feature {feature_id}: {len(has_feature)} representations")
        print(
            f"  |Cosine sim|: min={abs_sims.min():.4f}, max={abs_sims.max():.4f}, mean={abs_sims.mean():.4f}"
        )
        print(
            f"  Pairs above tau ({tau:.4f}): {(abs_sims >= tau).sum().item()}/{len(pairwise_sims)}"
        )
    print()

    clusters = cluster_by_neighbors(representations, tau)
    print(f"=== Clustering Results ===")
    print(f"Number of unique neighbor sets: {len(clusters)}")

    # Neighbor set size statistics
    neighbor_sizes = [len(ns) for ns in clusters.keys()]
    non_neighbor_sizes = [syn_config.num_representations - len(ns) for ns in clusters.keys()]
    print(f"Neighbor set sizes:     min={min(neighbor_sizes)}, max={max(neighbor_sizes)}, mean={sum(neighbor_sizes)/len(neighbor_sizes):.1f}")
    print(f"Non-neighbor set sizes: min={min(non_neighbor_sizes)}, max={max(non_neighbor_sizes)}, mean={sum(non_neighbor_sizes)/len(non_neighbor_sizes):.1f}")
    print()

    # Analyze each cluster
    print("=== Cluster Analysis ===")
    for idx, (neighbor_set, cluster_members) in enumerate(list(clusters.items())[:5]):
        if len(neighbor_set) < 2:
            continue

        pos_indices = list(neighbor_set)
        neg_indices = [
            i for i in range(syn_config.num_representations) if i not in neighbor_set
        ]

        # Count features in positive set
        feature_counts: dict[int, int] = {}
        for i in pos_indices:
            for f in repr_features[i]:
                feature_counts[f] = feature_counts.get(f, 0) + 1

        sorted_features = sorted(feature_counts.items(), key=lambda x: -x[1])
        dominant_feature = sorted_features[0][0]
        dominant_count = sorted_features[0][1]

        # Check negative set contamination
        neg_with_any_shared = sum(
            1 for i in neg_indices if repr_features[i] & set(feature_counts.keys())
        )
        neg_with_dominant = sum(
            1 for i in neg_indices if dominant_feature in repr_features[i]
        )

        # Extract feature and check alignment
        neighbor_indices = torch.tensor(pos_indices)
        nullspace = compute_nullspace(representations, neighbor_indices, 0.0, neg_tau=ext_config.neg_tau)
        if nullspace.shape[0] > 0:
            extracted = extract_feature(representations, neighbor_indices, nullspace)
            alignment = torch.abs(extracted @ features[dominant_feature]).item()
        else:
            alignment = 0.0

        print(f"Cluster {idx}: {len(cluster_members)} repr(s) share this neighbor set")
        print(
            f"  Neighbor set size: |pos|={len(pos_indices)}, |neg|={len(neg_indices)}"
        )
        print(f"  Feature counts: {dict(sorted_features[:5])}")
        print(
            f"  Dominant feature: {dominant_feature} (in {dominant_count}/{len(pos_indices)} = {100*dominant_count/len(pos_indices):.0f}%)"
        )
        print(
            f"  Neg with ANY shared feature: {neg_with_any_shared}/{len(neg_indices)} ({100*neg_with_any_shared/len(neg_indices):.0f}%)"
        )
        print(
            f"  Neg with DOMINANT feature: {neg_with_dominant}/{len(neg_indices)} ({100*neg_with_dominant/len(neg_indices):.0f}%)"
        )
        print(f"  Extracted alignment with dominant: {alignment:.4f}")
        print()

    # Check for duplicate extracted features
    print("=== Duplicate Analysis ===")
    extracted_features = []
    for neighbor_set, _ in clusters.items():
        if len(neighbor_set) < 2:
            continue
        neighbor_indices = torch.tensor(list(neighbor_set))
        try:
            nullspace = compute_nullspace(representations, neighbor_indices, 0.0, neg_tau=ext_config.neg_tau)
            if nullspace.shape[0] == 0:
                continue
            feature = extract_feature(representations, neighbor_indices, nullspace)
            extracted_features.append(feature)
        except ValueError:
            continue

    if len(extracted_features) > 1:
        feats = torch.stack(extracted_features)
        print(f"Extracted {len(extracted_features)} features (before dedup)")

        # Deduplicate
        sims = torch.abs(feats @ feats.T)
        keep_mask = torch.ones(len(feats), dtype=torch.bool)
        for i in range(len(feats)):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, len(feats)):
                if sims[i, j] > 0.99:
                    keep_mask[j] = False

        deduped = feats[keep_mask]
        print(f"After deduplication (0.99): {len(deduped)} features")

        # Check alignment with ground truth
        gt_sims = torch.abs(deduped @ features.T)
        best_per_extracted = gt_sims.max(dim=1).values
        best_per_gt = gt_sims.max(dim=0).values
        recovered = (best_per_gt > 0.9).sum().item()

        print(f"Ground truth recovered (align > 0.9): {recovered}/{syn_config.n}")
        print(f"Extracted alignment: min={best_per_extracted.min():.4f}, mean={best_per_extracted.mean():.4f}, max={best_per_extracted.max():.4f}")

        # Show how many spurious features (not aligned to any GT)
        spurious = (best_per_extracted < 0.9).sum().item()
        print(f"Spurious features (align < 0.9): {spurious}")


if __name__ == "__main__":
    main()
