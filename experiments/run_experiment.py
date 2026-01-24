import torch
import json
from pathlib import Path
from datetime import datetime

from src.config import ExperimentConfig, SyntheticConfig, ExtractionConfig
from src.synthetic import generate_feature_basis, generate_representations
from src.extraction import extract_all_features
from src.metrics import evaluate, MetricsResult
from src.logging_utils import log_experiment_config


def run_experiment(config: ExperimentConfig, verbose: bool = False) -> MetricsResult:
    """
    Execute full experiment pipeline.

    1. Set random seed for reproducibility
    2. Generate feature basis
    3. Generate representations with known coefficients
    4. Run extraction algorithm
    5. Evaluate metrics against ground truth
    6. Return results

    Args:
        config: ExperimentConfig with all settings
        verbose: If True, log detailed extraction information

    Returns:
        MetricsResult with evaluation metrics
    """
    if verbose:
        log_experiment_config(config)

    # Set seed
    torch.manual_seed(config.seed)

    # Generate ground truth
    basis_result = generate_feature_basis(
        config.synthetic.d,
        config.synthetic.n
    )
    features = basis_result.features

    representations, coefficients = generate_representations(
        features, config.synthetic
    )

    # Run extraction
    extracted = extract_all_features(
        representations, config.extraction, config.synthetic, verbose=verbose,
        use_minimality_filter=config.extraction.use_minimality_filter
    )

    # Evaluate
    result = evaluate(
        extracted=extracted,
        ground_truth=features,
        representations=representations,
        true_coefficients=coefficients,
        match_threshold=config.match_threshold
    )

    return result


def save_results(result: MetricsResult, config: ExperimentConfig, path: Path) -> None:
    """Save experiment results to JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "synthetic": {
                "d": config.synthetic.d,
                "n": config.synthetic.n,
                "epsilon": config.synthetic.epsilon,
                "num_representations": config.synthetic.num_representations,
                "sparsity_mode": config.synthetic.sparsity_mode,
                "k": config.synthetic.k,
            },
            "extraction": {
                "tau": config.extraction.tau,
                "epsilon": config.extraction.epsilon,
            },
            "seed": config.seed,
        },
        "results": {
            "recovery_rate": result.recovery_rate,
            "mean_alignment": result.mean_alignment,
            "reconstruction_error": result.reconstruction_error,
            "num_extracted": len(result.feature_matching),
            "num_unmatched_true": len(result.unmatched_true),
            "num_unmatched_extracted": len(result.unmatched_extracted),
        }
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run monosemantic feature extraction experiment")
    parser.add_argument("--d", type=int, default=64, help="Representation dimension")
    parser.add_argument("--n", type=int, default=64, help="Number of features")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Orthogonality tolerance")
    parser.add_argument("--num-repr", type=int, default=1000, help="Number of representations")
    parser.add_argument("--k", type=int, default=3, help="Sparsity (active features)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output path for results JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    config = ExperimentConfig(
        synthetic=SyntheticConfig(
            d=args.d,
            n=args.n,
            epsilon=args.epsilon,
            num_representations=args.num_repr,
            k=args.k,
        ),
        extraction=ExtractionConfig(tau=None, epsilon=args.epsilon),
        seed=args.seed,
    )

    print(f"Running experiment: d={args.d}, n={args.n}, epsilon={args.epsilon}, k={args.k}")
    result = run_experiment(config, verbose=args.verbose)

    print(f"\nResults:")
    print(f"  Recovery rate: {result.recovery_rate:.2%}")
    print(f"  Mean alignment: {result.mean_alignment:.4f}")
    print(f"  Reconstruction error: {result.reconstruction_error:.6f}")
    print(f"  Features extracted: {len(result.feature_matching)}")
    print(f"  Unmatched true: {len(result.unmatched_true)}")
    print(f"  Unmatched extracted: {len(result.unmatched_extracted)}")

    if args.output:
        save_results(result, config, Path(args.output))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
