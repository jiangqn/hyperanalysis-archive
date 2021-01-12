import torch
from hyperanalysis.utils import linalg

def scaled_covariance(X: torch.Tensor) -> torch.Tensor:
    """
    :param X: torch.FloatTensor (num, dim)
    :return C: torch.FloatTensor (dim, dim)
    """
    X_mean = X.mean(dim=0, keepdim=True)
    X = X - X_mean
    C = X.t().matmul(X)
    return C

def linear_discriminant_analysis(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    """
    :param X: torch.FloatTensor (num, dim)
    :param y: torch.LongTensor (num,)
    :return Z: torch.FloatTensor (num, 2)
    """

    assert len(X.size()) == 2
    assert len(y.size()) == 1
    assert X.size(0) == y.size(0)

    dtype = X.dtype
    device = X.device

    X = X - X.mean(dim=0, keepdim=True)

    num, dim = X.size()
    K = y.max().item() + 1

    St = scaled_covariance(X)
    Sw = torch.zeros(dim, dim, dtype=dtype, device=device)

    for k in range(K):

        index = torch.arange(num, device=X.device)
        mask = (y == k)
        # nk = mask.sum().item()
        index = index.masked_select(mask)
        Xk = X.index_select(dim=0, index=index)
        Sw = Sw + scaled_covariance(Xk)

    Sb = St - Sw

    Sb = Sb / num
    Sw = Sw / num

    if K == 2:

        pSw = linalg.postive_definite_matrix_power(Sw, -0.5)
        pSb = linalg.postive_definite_matrix_power(Sb, 0.5)

        B = pSb.matmul(pSw)
        _, s, V = torch.svd(B)

        sorted_indices = s.argsort(dim=0, descending=True)
        p = V[:, sorted_indices[0]]
        u = pSw.matmul(p)
        u = u / u.norm()
        v = linalg.constrained_max_variance(X, u)

        W = torch.stack([u, v], dim=1)
        Z = X.matmul(W)

    else: # K > 2

        pSw = linalg.postive_definite_matrix_power(Sw, -0.5)
        pSb = linalg.postive_definite_matrix_power(Sb, 0.5)

        B = pSb.matmul(pSw)
        _, s, V = torch.svd(B)

        sorted_indices = s.argsort(dim=0, descending=True)
        P = V[:, sorted_indices[0:2]]
        W = pSw.matmul(P)
        W = W / W.norm(dim=0, keepdim=True)

        Z = X.matmul(W)

    return Z