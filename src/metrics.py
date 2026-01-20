import torch
from dataclasses import dataclass


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
