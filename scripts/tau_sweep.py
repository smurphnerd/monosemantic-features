#!/usr/bin/env python3
"""
Tau sweep experiment: sweep tau threshold across synthetic configurations
and measure F1/precision/recall for identifying monosemantic targets.

Key question: Is there an optimal tau that can be predicted from known
generator parameters (epsilon, d/n ratio, Welch bound, mean similarity)?
"""

import json
import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.synthetic import generate_feature_basis, generate_representations, welch_bound
from src.extraction import build_neighbor_matrix, find_monosemantic_targets
from src.config import SyntheticConfig


@dataclass
class SweepConfig:
    """Configuration for a single sweep experiment."""
    name: str
    d: int
    n: int
    sparsity_mode: str
    k_mono: int       # k for monosemantic batch (always 1)
    k_poly: int       # k for polysemantic batch
    n_mono: int       # number of monosemantic representations
    n_poly: int       # number of polysemantic representations
    positive_only: bool = False
    seed: int = 42


def generate_mixed_data(cfg: SweepConfig):
    """Generate mixed monosemantic + polysemantic representations."""
    torch.manual_seed(cfg.seed)

    # Generate feature basis
    basis_result = generate_feature_basis(cfg.d, cfg.n)
    features = basis_result.features
    epsilon = basis_result.achieved_epsilon
    wb = welch_bound(cfg.n, cfg.d)

    # Generate monosemantic batch (k=1)
    mono_config = SyntheticConfig(
        d=cfg.d, n=cfg.n,
        num_representations=cfg.n_mono,
        sparsity_mode=cfg.sparsity_mode,
        k=1,
        coef_min_floor=0.1,
        coef_max=1.0,
        positive_only=cfg.positive_only,
    )
    mono_reps, mono_coeffs = generate_representations(features, mono_config, epsilon)

    # Generate polysemantic batch (k > 1)
    poly_config = SyntheticConfig(
        d=cfg.d, n=cfg.n,
        num_representations=cfg.n_poly,
        sparsity_mode=cfg.sparsity_mode,
        k=cfg.k_poly,
        coef_min_floor=0.1,
        coef_max=1.0,
        positive_only=cfg.positive_only,
    )
    poly_reps, poly_coeffs = generate_representations(features, poly_config, epsilon)

    # Concatenate
    all_reps = torch.cat([mono_reps, poly_reps], dim=0)
    all_coeffs = torch.cat([mono_coeffs, poly_coeffs], dim=0)

    # Ground truth: monosemantic = exactly 1 non-zero coefficient
    nonzero_counts = (all_coeffs.abs() > 1e-8).sum(dim=1)
    ground_truth_mono = (nonzero_counts == 1)

    return all_reps, all_coeffs, ground_truth_mono, features, epsilon, wb


def evaluate_tau(reps, ground_truth_mono, tau):
    """Evaluate a single tau value. Returns precision, recall, F1, n_targets."""
    neighbor_matrix = build_neighbor_matrix(reps, tau)
    target_indices = find_monosemantic_targets(neighbor_matrix)

    n_targets = len(target_indices)
    if n_targets == 0:
        return 0.0, 0.0, 0.0, 0

    # Predicted monosemantic mask
    predicted_mono = torch.zeros(len(reps), dtype=torch.bool)
    predicted_mono[target_indices] = True

    # Compute metrics
    tp = (predicted_mono & ground_truth_mono).sum().item()
    fp = (predicted_mono & ~ground_truth_mono).sum().item()
    fn = (~predicted_mono & ground_truth_mono).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, n_targets


def compute_mean_similarity(reps):
    """Compute mean pairwise absolute cosine similarity."""
    norms = torch.norm(reps, dim=1, keepdim=True)
    X_norm = reps / (norms + 1e-8)
    sims = torch.abs(X_norm @ X_norm.T)
    # Exclude diagonal
    n = sims.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    return sims[mask].mean().item()


def run_sweep(cfg: SweepConfig, tau_values):
    """Run tau sweep for one configuration."""
    print(f"\n{'='*60}")
    print(f"Config: {cfg.name}")
    print(f"  d={cfg.d}, n={cfg.n}, sparsity={cfg.sparsity_mode}, k_poly={cfg.k_poly}")
    print(f"  n_mono={cfg.n_mono}, n_poly={cfg.n_poly}")

    reps, coeffs, gt_mono, features, epsilon, wb = generate_mixed_data(cfg)
    mean_sim = compute_mean_similarity(reps)
    dn_ratio = cfg.d / cfg.n

    print(f"  epsilon={epsilon:.4f}, welch_bound={wb:.4f}, mean_sim={mean_sim:.4f}")
    print(f"  d/n ratio={dn_ratio:.3f}")
    print(f"  Total reps={len(reps)}, true mono={gt_mono.sum().item()}")

    results = []
    for tau in tau_values:
        precision, recall, f1, n_targets = evaluate_tau(reps, gt_mono, tau)
        results.append({
            'tau': float(tau),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_targets': n_targets,
        })

    # Find optimal tau (by F1)
    best = max(results, key=lambda r: r['f1'])
    print(f"  Best tau={best['tau']:.3f} → F1={best['f1']:.3f}, P={best['precision']:.3f}, R={best['recall']:.3f}, targets={best['n_targets']}")

    return {
        'config': cfg.name,
        'd': cfg.d,
        'n': cfg.n,
        'sparsity_mode': cfg.sparsity_mode,
        'k_poly': cfg.k_poly,
        'n_mono': cfg.n_mono,
        'n_poly': cfg.n_poly,
        'epsilon': epsilon,
        'welch_bound': wb,
        'mean_similarity': mean_sim,
        'dn_ratio': dn_ratio,
        'n_true_mono': int(gt_mono.sum().item()),
        'optimal_tau': best['tau'],
        'optimal_f1': best['f1'],
        'optimal_precision': best['precision'],
        'optimal_recall': best['recall'],
        'sweep': results,
    }


def plot_sweep_results(all_results, output_dir):
    """Generate plots for all sweep results."""

    # Plot 1: F1/precision/recall curves per config
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, result in enumerate(all_results):
        if i >= len(axes):
            break
        ax = axes[i]
        taus = [r['tau'] for r in result['sweep']]
        f1s = [r['f1'] for r in result['sweep']]
        precs = [r['precision'] for r in result['sweep']]
        recs = [r['recall'] for r in result['sweep']]
        n_tgts = [r['n_targets'] for r in result['sweep']]

        ax.plot(taus, f1s, 'b-', linewidth=2, label='F1')
        ax.plot(taus, precs, 'g--', linewidth=1.5, label='Precision')
        ax.plot(taus, recs, 'r--', linewidth=1.5, label='Recall')

        # Mark optimal
        opt_tau = result['optimal_tau']
        opt_f1 = result['optimal_f1']
        ax.axvline(opt_tau, color='blue', alpha=0.3, linestyle=':')
        ax.plot(opt_tau, opt_f1, 'b*', markersize=12)

        # Mark epsilon and welch bound
        if result['epsilon'] > 0:
            ax.axvline(result['epsilon'], color='orange', alpha=0.5, linestyle='--', label=f'ε={result["epsilon"]:.3f}')
        if result['welch_bound'] > 0:
            ax.axvline(result['welch_bound'], color='purple', alpha=0.5, linestyle='--', label=f'WB={result["welch_bound"]:.3f}')

        ax.set_title(f"{result['config']}\nopt τ={opt_tau:.3f}, F1={opt_f1:.3f}", fontsize=9)
        ax.set_xlabel('τ')
        ax.set_ylabel('Score')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(len(all_results), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tau_sweep_curves.png'), dpi=150)
    plt.close()
    print(f"\nSaved: {output_dir}/tau_sweep_curves.png")

    # Plot 2: Optimal tau vs predictors
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    opt_taus = [r['optimal_tau'] for r in all_results]
    epsilons = [r['epsilon'] for r in all_results]
    dn_ratios = [r['dn_ratio'] for r in all_results]
    wbs = [r['welch_bound'] for r in all_results]
    mean_sims = [r['mean_similarity'] for r in all_results]
    labels = [r['config'] for r in all_results]

    for ax, xs, xlabel in zip(axes, [epsilons, dn_ratios, wbs, mean_sims],
                               ['ε (achieved)', 'd/n ratio', 'Welch bound', 'Mean |cos sim|']):
        ax.scatter(xs, opt_taus, s=80, zorder=5)
        for x, y, lbl in zip(xs, opt_taus, labels):
            ax.annotate(lbl, (x, y), fontsize=6, rotation=15, ha='left', va='bottom')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Optimal τ')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Optimal τ vs Generator Parameters', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tau_sweep_predictors.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/tau_sweep_predictors.png")

    # Plot 3: Number of targets found vs tau
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, result in enumerate(all_results):
        if i >= len(axes):
            break
        ax = axes[i]
        taus = [r['tau'] for r in result['sweep']]
        n_tgts = [r['n_targets'] for r in result['sweep']]
        true_mono = result['n_true_mono']

        ax.plot(taus, n_tgts, 'k-', linewidth=2, label='Targets found')
        ax.axhline(true_mono, color='green', linestyle='--', alpha=0.7, label=f'True mono={true_mono}')
        ax.axvline(result['optimal_tau'], color='blue', alpha=0.3, linestyle=':', label=f'opt τ={result["optimal_tau"]:.3f}')

        ax.set_title(result['config'], fontsize=9)
        ax.set_xlabel('τ')
        ax.set_ylabel('# Targets')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(len(all_results), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tau_sweep_targets.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/tau_sweep_targets.png")


def main():
    output_dir = Path(__file__).resolve().parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)

    # Define configurations
    configs = [
        # Fixed sparsity, k_poly=2
        SweepConfig("d16_n16_fixed_k2", d=16, n=16, sparsity_mode="fixed", k_mono=1, k_poly=2, n_mono=80, n_poly=120),
        SweepConfig("d16_n20_fixed_k2", d=16, n=20, sparsity_mode="fixed", k_mono=1, k_poly=2, n_mono=100, n_poly=150),
        SweepConfig("d16_n24_fixed_k2", d=16, n=24, sparsity_mode="fixed", k_mono=1, k_poly=2, n_mono=120, n_poly=180),
        SweepConfig("d32_n48_fixed_k2", d=32, n=48, sparsity_mode="fixed", k_mono=1, k_poly=2, n_mono=200, n_poly=300),

        # Bernoulli-Gaussian, k_poly=3
        SweepConfig("d16_n16_bg_k3", d=16, n=16, sparsity_mode="bernoulli_gaussian", k_mono=1, k_poly=3, n_mono=80, n_poly=120),
        SweepConfig("d16_n20_bg_k3", d=16, n=20, sparsity_mode="bernoulli_gaussian", k_mono=1, k_poly=3, n_mono=100, n_poly=150),
        SweepConfig("d16_n24_bg_k3", d=16, n=24, sparsity_mode="bernoulli_gaussian", k_mono=1, k_poly=3, n_mono=120, n_poly=180),
        SweepConfig("d32_n48_bg_k3", d=32, n=48, sparsity_mode="bernoulli_gaussian", k_mono=1, k_poly=3, n_mono=200, n_poly=300),
    ]

    # Tau values to sweep
    tau_values = np.linspace(0.01, 0.95, 50)

    all_results = []
    for cfg in configs:
        result = run_sweep(cfg, tau_values)
        all_results.append(result)

    # Summary table
    print(f"\n{'='*120}")
    print("SUMMARY TABLE")
    print(f"{'='*120}")
    header = f"{'Config':<25} {'ε':>7} {'WB':>7} {'d/n':>6} {'Mean sim':>9} {'Opt τ':>7} {'F1':>6} {'Prec':>6} {'Rec':>6} {'#Tgt':>5}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['config']:<25} {r['epsilon']:>7.4f} {r['welch_bound']:>7.4f} {r['dn_ratio']:>6.3f} "
              f"{r['mean_similarity']:>9.4f} {r['optimal_tau']:>7.3f} {r['optimal_f1']:>6.3f} "
              f"{r['optimal_precision']:>6.3f} {r['optimal_recall']:>6.3f} {r['n_true_mono']:>5d}")

    # Correlation analysis
    print(f"\n{'='*60}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*60}")
    opt_taus = np.array([r['optimal_tau'] for r in all_results])
    for name, vals in [
        ('epsilon', [r['epsilon'] for r in all_results]),
        ('welch_bound', [r['welch_bound'] for r in all_results]),
        ('d/n ratio', [r['dn_ratio'] for r in all_results]),
        ('mean_similarity', [r['mean_similarity'] for r in all_results]),
    ]:
        vals = np.array(vals)
        if np.std(vals) > 1e-10 and np.std(opt_taus) > 1e-10:
            corr = np.corrcoef(vals, opt_taus)[0, 1]
            print(f"  Corr(optimal_tau, {name:>15s}) = {corr:+.4f}")
        else:
            print(f"  Corr(optimal_tau, {name:>15s}) = N/A (no variance)")

    # Save results
    output_path = output_dir / 'tau_sweep_results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_path}")

    # Generate plots
    plot_sweep_results(all_results, str(output_dir))

    print("\nDone!")


if __name__ == '__main__':
    main()
