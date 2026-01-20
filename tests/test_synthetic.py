import pytest
import torch
from src.synthetic import generate_feature_basis


def test_generate_orthogonal_basis_shape():
    features = generate_feature_basis(d=64, n=64, epsilon=0.0)
    assert features.shape == (64, 64)


def test_generate_orthogonal_basis_unit_norm():
    features = generate_feature_basis(d=64, n=64, epsilon=0.0)
    norms = torch.norm(features, dim=1)
    assert torch.allclose(norms, torch.ones(64), atol=1e-6)


def test_generate_orthogonal_basis_orthogonality():
    features = generate_feature_basis(d=64, n=64, epsilon=0.0)
    # Gram matrix should be identity for orthonormal basis
    gram = features @ features.T
    identity = torch.eye(64)
    assert torch.allclose(gram, identity, atol=1e-6)


def test_generate_orthogonal_basis_requires_n_leq_d():
    with pytest.raises(ValueError, match="n <= d"):
        generate_feature_basis(d=32, n=64, epsilon=0.0)


from src.config import SyntheticConfig
from src.synthetic import generate_representations


def test_generate_representations_shape():
    config = SyntheticConfig(d=64, n=64, epsilon=0.0, num_representations=100, k=3)
    features = generate_feature_basis(config.d, config.n, config.epsilon)
    representations, coefficients = generate_representations(features, config)

    assert representations.shape == (100, 64)
    assert coefficients.shape == (100, 64)


def test_generate_representations_fixed_sparsity():
    config = SyntheticConfig(
        d=64, n=64, epsilon=0.0, num_representations=100,
        sparsity_mode="fixed", k=3
    )
    features = generate_feature_basis(config.d, config.n, config.epsilon)
    _, coefficients = generate_representations(features, config)

    # Each representation should have exactly k=3 non-zero coefficients
    nonzero_counts = (coefficients != 0).sum(dim=1)
    assert torch.all(nonzero_counts == 3)


def test_generate_representations_coefficient_bounds():
    config = SyntheticConfig(
        d=64, n=64, epsilon=0.0, num_representations=100,
        k=3, coef_min_floor=0.1, coef_max=1.0
    )
    features = generate_feature_basis(config.d, config.n, config.epsilon)
    _, coefficients = generate_representations(features, config)

    # Non-zero coefficients should have |c| in [0.1, 1.0]
    nonzero_mask = coefficients != 0
    nonzero_abs = torch.abs(coefficients[nonzero_mask])
    assert torch.all(nonzero_abs >= 0.1)
    assert torch.all(nonzero_abs <= 1.0)


def test_generate_representations_reconstruction():
    config = SyntheticConfig(d=64, n=64, epsilon=0.0, num_representations=100, k=3)
    features = generate_feature_basis(config.d, config.n, config.epsilon)
    representations, coefficients = generate_representations(features, config)

    # representations should equal coefficients @ features
    reconstructed = coefficients @ features
    assert torch.allclose(representations, reconstructed, atol=1e-6)
