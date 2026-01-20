import torch


def generate_feature_basis(d: int, n: int, epsilon: float) -> torch.Tensor:
    """
    Generate n feature vectors in d dimensions that are ε-orthogonal.

    Args:
        d: Dimension of representation space
        n: Number of features
        epsilon: Orthogonality tolerance (|⟨f_i, f_j⟩| ≤ ε for i≠j)

    Returns:
        (n, d) tensor of unit-norm feature vectors

    For ε=0: Returns orthonormal basis (requires n ≤ d)
    For ε>0: Not yet implemented
    """
    if epsilon == 0.0:
        if n > d:
            raise ValueError(f"For epsilon=0, requires n <= d, got n={n}, d={d}")
        # Generate random matrix and orthogonalize via QR
        random_matrix = torch.randn(d, n)
        q, _ = torch.linalg.qr(random_matrix)
        # q is (d, n), we want (n, d)
        return q.T
    else:
        raise NotImplementedError(f"epsilon > 0 not yet implemented, got epsilon={epsilon}")
