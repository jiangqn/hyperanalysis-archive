import torch
from torch import nn
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

class LinearDiscriminantAnalysis(object):

    def __init__(self, n_components: int = None) -> None:
        super(LinearDiscriminantAnalysis, self).__init__()
        self.n_components = n_components

    def fit(self, X: torch.FloatTensor, y: torch.LongTensor) -> None:
        self._fit(X, y)

    def transform(self, X: torch.FloatTensor) -> torch.FloatTensor:
        return self._transform(X)

    def fit_transform(self, X: torch.FloatTensor, y: torch.LongTensor) -> torch.FloatTensor:
        self._fit(X, y)
        return self._transform(X)

    def _fit(self, X: torch.FloatTensor, y: torch.LongTensor) -> None:

        X, y = X.clone(), y.clone()

        assert len(X.size()) == 2
        assert len(y.size()) == 1
        assert X.size(0) == y.size(0)

        dtype = X.dtype
        device = X.device

        self.mean_ = X.mean(dim=0, keepdim=True)
        X -= self.mean_

        num, dim = X.size()
        K = y.max().item() + 1

        assert self.n_components <= min(dim, K - 1)

        St = scaled_covariance(X)
        Sw = torch.zeros(dim, dim, dtype=dtype, device=device)

        for k in range(K):
            index = torch.arange(num, device=X.device)
            mask = (y == k)
            index = index.masked_select(mask)
            Xk = X.index_select(dim=0, index=index)
            Sw = Sw + scaled_covariance(Xk)

        Sb = St - Sw

        Sb = Sb / num
        Sw = Sw / num

        pSw = linalg.postive_definite_matrix_power(Sw, -0.5)
        pSb = linalg.postive_definite_matrix_power(Sb, 0.5)

        B = pSb.matmul(pSw)
        _, s, V = torch.svd(B)

        sorted_indices = s.argsort(dim=0, descending=True)
        P = V[:, sorted_indices[0:self.n_components]]
        W = pSw.matmul(P)
        W = W / W.norm(dim=0, keepdim=True)

        self.W_ = W

    def _transform(self, X: torch.FloatTensor) -> torch.FloatTensor:

        assert hasattr(self, "mean_")
        assert hasattr(self, "W_")

        return (X - self.mean_).matmul(self.W_)