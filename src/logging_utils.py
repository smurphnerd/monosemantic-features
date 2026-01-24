import logging

import torch

from src.config import ExperimentConfig


def get_logger(name: str = "monosemantic") -> logging.Logger:
    """Get or create a logger with the given name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger()


def log_experiment_config(config: ExperimentConfig) -> None:
    """Log experiment hyperparameters, filtering by relevance."""
    lines = ["Experiment Configuration:"]

    # Synthetic config - conditionally include fields
    syn = config.synthetic
    lines.append(f"  Synthetic: d={syn.d}, n={syn.n}, num_repr={syn.num_representations}")
    lines.append(f"    sparsity_mode={syn.sparsity_mode}, k={syn.k}", )
    if syn.sparsity_mode == "variable" and syn.k_min is not None:
        lines.append(f"    k_min={syn.k_min}")
    lines.append(f"    coef_max={syn.coef_max}, coef_min_floor={syn.coef_min_floor}")

    # Extraction config - conditionally include fields
    ext = config.extraction
    if ext.tau is not None:
        lines.append(f"  Extraction: tau={ext.tau}")
    else:
        lines.append(f"  Extraction: tau=auto (margin={ext.tau_margin})")
    if ext.epsilon > 0:
        lines.append(f"    epsilon={ext.epsilon}")
    if ext.max_neighbors is not None:
        lines.append(f"    max_neighbors={ext.max_neighbors}")

    # Experiment-level
    lines.append(f"  seed={config.seed}, match_threshold={config.match_threshold}")

    logger.info("\n".join(lines))


def log_neighbor_segmentation(
    target_idx: int,
    found_neighbors: torch.Tensor,
    true_feature_indices: dict[int, set[int]] | None = None,
) -> None:
    """
    Log neighbor finding results with optional ground truth comparison.

    Args:
        target_idx: Index of the target representation
        found_neighbors: Tensor of neighbor indices
        true_feature_indices: Optional mapping from feature_id -> set of repr indices
                              that have that feature active
    """
    n_found = len(found_neighbors)

    if true_feature_indices is None:
        logger.info(f"Neighbors for repr {target_idx}: found {n_found}")
        return

    # Find which features the target has
    target_features = [
        fid for fid, indices in true_feature_indices.items()
        if target_idx in indices
    ]

    if not target_features:
        logger.info(f"Neighbors for repr {target_idx}: found {n_found} (no ground truth features)")
        return

    # True neighbors: representations sharing at least one feature with target
    true_neighbors = set()
    for fid in target_features:
        true_neighbors.update(true_feature_indices[fid])

    found_set = set(found_neighbors.tolist())

    correct = len(found_set & true_neighbors)
    false_positives = len(found_set - true_neighbors)
    false_negatives = len(true_neighbors - found_set)

    lines = [
        f"Neighbors for repr {target_idx}:",
        f"  found={n_found}, true={len(true_neighbors)}",
        f"  correct={correct}, false_pos={false_positives}, false_neg={false_negatives}",
    ]
    logger.info("\n".join(lines))


def log_nullspace_computation(
    n_neg: int,
    rms_norm: float,
    epsilon: float,
    scaled_epsilon: float,
    singular_values: torch.Tensor,
    nullspace_dim: int,
) -> None:
    """
    Log nullspace computation details.

    Args:
        n_neg: Number of non-neighbor representations
        rms_norm: RMS of representation norms
        epsilon: Raw epsilon parameter
        scaled_epsilon: Computed epsilon tilde
        singular_values: Full singular value tensor (padded to d)
        nullspace_dim: Number of singular values below threshold
    """
    # Find first singular value below threshold
    below_mask = singular_values < scaled_epsilon
    first_below = None
    if below_mask.any():
        first_below_idx = int(torch.where(below_mask)[0][0])
        first_below = float(singular_values[first_below_idx])

    lines = [
        f"Nullspace computation:",
        f"  n_neg={n_neg}, rms_norm={rms_norm:.4f}",
        f"  epsilon_tilde = sqrt({n_neg}) * {rms_norm:.4f} * {epsilon} = {scaled_epsilon:.6f}",
        f"  singular values below threshold: {nullspace_dim}",
    ]
    if first_below is not None:
        lines.append(f"  first SV below threshold: {first_below:.6f}")

    logger.info("\n".join(lines))


def log_feature_extraction(
    n_neighbors: int,
    singular_values: torch.Tensor,
) -> None:
    """
    Log feature extraction SVD results.

    Args:
        n_neighbors: Number of neighbor representations projected
        singular_values: Singular values from projected neighbors SVD
    """
    first_sv = float(singular_values[0]) if len(singular_values) > 0 else 0.0
    second_sv = float(singular_values[1]) if len(singular_values) > 1 else 0.0

    lines = [
        f"Feature extraction:",
        f"  n_neighbors={n_neighbors}",
        f"  first SV={first_sv:.4f}, second SV={second_sv:.4f}",
        f"  ratio={first_sv / (second_sv + 1e-8):.2f}",
    ]
    logger.info("\n".join(lines))


def log_cluster_summary(
    num_clusters: int,
    cluster_sizes: list[int],
) -> None:
    """Log summary of clustering results."""
    if not cluster_sizes:
        logger.info("Clustering: no clusters formed")
        return

    lines = [
        f"Clustering: {num_clusters} clusters",
        f"  sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
        f"mean={sum(cluster_sizes)/len(cluster_sizes):.1f}",
    ]
    logger.info("\n".join(lines))


def log_deduplication(
    before_count: int,
    after_count: int,
) -> None:
    """Log deduplication results."""
    removed = before_count - after_count
    if removed > 0:
        logger.info(f"Deduplication: {before_count} -> {after_count} features ({removed} duplicates removed)")
