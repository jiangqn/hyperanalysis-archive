import torch
from hyperanalysis.utils import linalg

def linear_regression_analysis(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    """
    :param X: torch.FloatTensor (num, dim)
    :param y: torch.FloatTensor (num,)
    :return Z: torch.FloatTensor (num, 2)
    """

    assert len(X.size()) == 2
    assert len(y.size()) == 1
    assert X.size(0) == y.size(0)

    X = X - X.mean(dim=0, keepdim=True)

    num, dim = X.size()
    ones = torch.ones(size=(num, 1), dtype=X.dtype, device=X.device)
    X = torch.cat((X, ones), dim=1)
    Y = y.unsqueeze(-1)

    U = torch.inverse(X.t().matmul(X)).matmul(X.t()).matmul(Y)

    X = X[:, 0:-1]
    u = U[0:-1, 0]
    u = u / u.norm()
    v = linalg.constrained_max_variance(X, u)
    W = torch.stack([u, v], dim=1)
    Z = X.matmul(W)

    return Z