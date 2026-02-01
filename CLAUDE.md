# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research testing infrastructure for validating monosemantic feature extraction. The core hypothesis: neural network representations are sparse linear combinations of nearly-orthogonal "monosemantic features" (each responding to a single semantic concept). This codebase generates synthetic data with known ground truth, extracts features using neighbor-based clustering and nullspace analysis, and evaluates recovery accuracy.

## Commands

```bash
# Install (uses mamba/conda)
mamba create -n monosemantic python=3.11 -y
mamba activate monosemantic
pip install -e ".[dev]"

# Run all tests
pytest

# Run single test file
pytest tests/test_extraction.py

# Run single test
pytest tests/test_extraction.py::test_find_tau_neighbors_basic -v

# Run experiments
python experiments/run_experiment.py --d 16 --n 16 --k 2 --num-repr 20
python experiments/run_experiment.py --d 16 --n 16 --k 2 --num-repr 20 --output results/experiment.json
```

## Architecture

### Extraction Algorithm Pipeline

1. **Neighbor Discovery** (`find_tau_neighbors`): Groups representations by cosine similarity ≥ τ (threshold auto-derived from sparsity parameters or manually set). Representations sharing features cluster together.

2. **Target Selection** (`select_monosemantic_targets`): Identifies extraction targets - either all unique neighbor groups or only those with locally-minimal neighbor counts (cleaner single-feature representations).

3. **Feature Extraction** (`extract_features_for_target`): For each target, computes ε-nullspace of non-neighbor representations, projects neighbors onto nullspace, extracts dominant direction via SVD.

4. **Deduplication**: Removes near-duplicate extracted features (cosine similarity > 0.99).

### Key Modules

- **`src/synthetic.py`**: Generates ε-orthogonal feature bases (QR for n≤d, Gram matrix alternating projection for n>d approaching Welch bound) and sparse representation mixtures.

- **`src/extraction.py`**: Core extraction algorithm. Auto-derives τ threshold bounds from generative parameters.

- **`src/metrics.py`**: Greedy feature matching and recovery rate computation.

- **`src/config.py`**: Three config levels: `SyntheticConfig` (data generation), `ExtractionConfig` (algorithm params), `ExperimentConfig` (combined).

### Conventions

- **Row-major matrices**: Representations and features are row vectors
- **Unit-norm features**: Enforced throughout via `F.normalize` or SVD
- **Absolute cosine similarity**: Sign-invariant matching (features can point either direction)

## Testing Philosophy

- **Unit test all functions** - Every function should have unit tests covering its behavior
- **No arbitrary metric goals in tests** - Avoid tests that check for specific numeric thresholds like "should achieve within 2x of Welch bound"
- **Skip integration tests with soft goals** - The researcher will run experiments to validate research ideas work end-to-end
- **Test behavior, not benchmarks** - Test that functions return correct types, shapes, and satisfy mathematical invariants (e.g., unit norm, positive values where expected)

## Worktrees

Worktree directory: `.worktrees/` (gitignored)

## Known Constraints

Nullspace-based extraction requires n_non_neighbors < d. For d=16, use num_representations ≤ 20.
