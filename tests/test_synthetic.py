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
