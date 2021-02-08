import torch
from hyperanalysis.decomposition.base import UnsupervisedDecomposition

class PCA(UnsupervisedDecomposition):

    def __init__(self, n_components: int = 2) -> None:
        super(PCA, self).__init__(n_components)

    def _fit(self, X: torch.Tensor) -> None:
        num, dim = X.size()
        assert num >= 2

        self._mean = X.mean(dim=0, keepdim=True)
        X = X - self._mean

        _, S, V = torch.svd(X)

        index = S.argsort(descending=True)
        S = S.index_select(index=index, dim=0)
        V = V.index_select(index=index, dim=1)

        eigen_values = S * S / (num - 1)

        self._weight = V[:, 0:self.n_components]
        self._explained_variance_ratio = eigen_values[0:self.n_components] / eigen_values.sum()

    def _transform(self, X: torch.FloatTensor) -> torch.FloatTensor:
        return (X - self._mean).matmul(self._weight)

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def weight(self) -> torch.Tensor:
        return self._weight

    @property
    def explained_variance_ratio(self) -> torch.Tensor:
        return self._explained_variance_ratio