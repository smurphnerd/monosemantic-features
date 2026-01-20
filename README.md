# monosemantic-features

Testing infrastructure for validating monosemantic feature extraction methodology.

## Installation

```bash
mamba create -n monosemantic python=3.11 -y
mamba activate monosemantic
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest -v
```

## Running Experiments

```bash
# Basic orthogonal experiment
python experiments/run_experiment.py --d 16 --n 16 --k 2 --num-repr 20

# Save results
python experiments/run_experiment.py --d 16 --n 16 --k 2 --num-repr 20 --output results/experiment.json
```

## Project Structure

- `src/config.py` - Configuration dataclasses
- `src/synthetic.py` - Feature basis and representation generation
- `src/extraction.py` - Feature extraction algorithm
- `src/metrics.py` - Evaluation metrics
- `experiments/run_experiment.py` - Experiment runner CLI
- `docs/plans/` - Design documents and implementation plans
- `writeup/` - CVPR 2025 paper

## Methodology

See `docs/plans/2026-01-20-testing-infrastructure-design.md` for detailed design.

## Known Limitations

The nullspace-based extraction requires that the number of non-neighbor representations
is less than the dimension d. With many representations, the nullspace becomes trivial.
For d=16, use num_representations ≤ 20 for best results.
