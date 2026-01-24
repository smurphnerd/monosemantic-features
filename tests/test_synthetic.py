import pytest
import torch
from src.synthetic import generate_feature_basis


def test_generate_orthogonal_basis_shape():
    result = generate_feature_basis(d=10, n=5)
    assert result.features.shape == (5, 10)


def test_generate_orthogonal_basis_unit_norm():
    result = generate_feature_basis(d=10, n=5)
    norms = torch.linalg.norm(result.features, dim=1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-6)


def test_generate_orthogonal_basis_orthogonality():
    result = generate_feature_basis(d=10, n=5)
    G = result.features @ result.features.T
    # Off-diagonal should be ~0
    off_diag = G - torch.eye(5)
    assert torch.allclose(off_diag, torch.zeros_like(off_diag), atol=1e-6)


from src.config import SyntheticConfig
from src.synthetic import generate_representations


def test_generate_representations_shape():
    config = SyntheticConfig(d=64, n=64, num_representations=100, k=3)
    result = generate_feature_basis(config.d, config.n)
    representations, coefficients = generate_representations(result.features, config)

    assert representations.shape == (100, 64)
    assert coefficients.shape == (100, 64)


def test_generate_representations_fixed_sparsity():
    config = SyntheticConfig(
        d=64, n=64, num_representations=100,
        sparsity_mode="fixed", k=3
    )
    result = generate_feature_basis(config.d, config.n)
    _, coefficients = generate_representations(result.features, config)

    # Each representation should have exactly k=3 non-zero coefficients
    nonzero_counts = (coefficients != 0).sum(dim=1)
    assert torch.all(nonzero_counts == 3)


def test_generate_representations_coefficient_bounds():
    config = SyntheticConfig(
        d=64, n=64, num_representations=100,
        k=3, coef_min_floor=0.1, coef_max=1.0
    )
    result = generate_feature_basis(config.d, config.n)
    _, coefficients = generate_representations(result.features, config)

    # Non-zero coefficients should have |c| in [0.1, 1.0]
    nonzero_mask = coefficients != 0
    nonzero_abs = torch.abs(coefficients[nonzero_mask])
    assert torch.all(nonzero_abs >= 0.1)
    assert torch.all(nonzero_abs <= 1.0)


def test_generate_representations_reconstruction():
    config = SyntheticConfig(d=64, n=64, num_representations=100, k=3)
    result = generate_feature_basis(config.d, config.n)
    representations, coefficients = generate_representations(result.features, config)

    # representations should equal coefficients @ features
    reconstructed = coefficients @ result.features
    assert torch.allclose(representations, reconstructed, atol=1e-6)


def test_welch_bound_orthogonal_possible():
    """When n <= d, Welch bound is 0 (orthonormal basis possible)."""
    from src.synthetic import welch_bound
    assert welch_bound(n=5, d=10) == 0.0
    assert welch_bound(n=10, d=10) == 0.0


def test_welch_bound_overcomplete():
    """When n > d, Welch bound is positive."""
    from src.synthetic import welch_bound
    # n=100, d=64: sqrt((100-64)/(64*(100-1))) = sqrt(36/6336) ≈ 0.0754
    bound = welch_bound(n=100, d=64)
    assert bound > 0
    assert abs(bound - 0.0754) < 0.001


def test_welch_bound_formula():
    """Verify exact formula: sqrt((n-d)/(d*(n-1)))."""
    import math
    from src.synthetic import welch_bound
    n, d = 20, 10
    expected = math.sqrt((n - d) / (d * (n - 1)))
    assert abs(welch_bound(n, d) - expected) < 1e-10


def test_feature_basis_result_dataclass():
    """FeatureBasisResult holds features and metadata."""
    import torch
    from src.synthetic import FeatureBasisResult

    features = torch.randn(5, 3)
    result = FeatureBasisResult(
        features=features,
        achieved_epsilon=0.1,
        welch_bound=0.05,
        converged=True
    )

    assert result.features.shape == (5, 3)
    assert result.achieved_epsilon == 0.1
    assert result.welch_bound == 0.05
    assert result.converged is True


def test_generate_feature_basis_returns_result():
    """generate_feature_basis returns FeatureBasisResult."""
    from src.synthetic import generate_feature_basis, FeatureBasisResult

    result = generate_feature_basis(d=5, n=3)

    assert isinstance(result, FeatureBasisResult)
    assert result.features.shape == (3, 5)
    assert result.achieved_epsilon == 0.0
    assert result.welch_bound == 0.0
    assert result.converged is True


def test_generate_feature_basis_orthonormal():
    """For n <= d, features are orthonormal."""
    import torch
    from src.synthetic import generate_feature_basis

    result = generate_feature_basis(d=5, n=5)
    F = result.features

    # Unit norm
    norms = torch.linalg.norm(F, dim=1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-6)

    # Orthogonal (G = I for orthonormal)
    G = F @ F.T
    assert torch.allclose(G, torch.eye(5), atol=1e-6)


def test_generate_feature_basis_overcomplete_shape():
    """For n > d, still returns correct shape."""
    from src.synthetic import generate_feature_basis

    result = generate_feature_basis(d=5, n=10)

    assert result.features.shape == (10, 5)


def test_generate_feature_basis_overcomplete_unit_norm():
    """For n > d, features are unit norm."""
    import torch
    from src.synthetic import generate_feature_basis

    result = generate_feature_basis(d=5, n=10)
    norms = torch.linalg.norm(result.features, dim=1)

    assert torch.allclose(norms, torch.ones(10), atol=1e-5)


def test_generate_feature_basis_overcomplete_coherence():
    """For n > d, achieved_epsilon >= welch_bound (can't beat physics)."""
    from src.synthetic import generate_feature_basis

    result = generate_feature_basis(d=5, n=10)

    assert result.achieved_epsilon >= result.welch_bound - 1e-6
    assert result.welch_bound > 0  # Overcomplete case


def test_generate_feature_basis_overcomplete_coherence_matches():
    """achieved_epsilon matches actual max coherence."""
    import torch
    from src.synthetic import generate_feature_basis

    result = generate_feature_basis(d=5, n=10)
    F = result.features

    G = F @ F.T
    G.fill_diagonal_(0)
    actual_max = G.abs().max().item()

    assert abs(actual_max - result.achieved_epsilon) < 1e-6
