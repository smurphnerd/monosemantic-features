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
