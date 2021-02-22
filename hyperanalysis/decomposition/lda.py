import torch
from hyperanalysis import linalg
from hyperanalysis.decomposition.base import SupervisedDecomposition

def scaled_covariance(X: torch.Tensor) -> torch.Tensor:
    """
    :param X: torch.FloatTensor (num, dim)
    :return C: torch.FloatTensor (dim, dim)
    """
    X_mean = X.mean(dim=0, keepdim=True)
    X = X - X_mean
    C = X.t().matmul(X)
    return C

class LinearDiscriminantAnalysis(SupervisedDecomposition):

    def __init__(self, n_components: int = 1) -> None:
        super(LinearDiscriminantAnalysis, self).__init__(n_components)

    def _fit(self, X: torch.Tensor, y: torch.LongTensor) -> None:

        dtype = X.dtype
        device = X.device

        self._mean = X.mean(dim=0, keepdim=True)
        X = X - self._mean

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
        weight = pSw.matmul(P)
        weight = weight / weight.norm(dim=0, keepdim=True)

        self._weight = weight

    def _transform(self, X: torch.FloatTensor) -> torch.FloatTensor:
        return (X - self._mean).matmul(self._weight)

    @property
    def mean(self) -> torch.Tensor:
        assert self.is_trained
        return self._mean

    @property
    def weight(self) -> torch.Tensor:
        assert self.is_trained
        return self._weight