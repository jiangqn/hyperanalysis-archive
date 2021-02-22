import torch
from hyperanalysis.decomposition.base import UnsupervisedDecomposition

class TruncatedSVD(UnsupervisedDecomposition):

    def __init__(self, n_components: int = 2) -> None:
        super(TruncatedSVD, self).__init__(n_components)

    def _fit(self, X: torch.Tensor) -> None:
        num, dim = X.size()
        assert num >= 2

        _, S, V = torch.svd(X)

        index = S.argsort(descending=True)
        S = S.index_select(index=index, dim=0)
        V = V.index_select(index=index, dim=1)

        eigen_values = S * S / (num - 1)

        self._weight = V[:, 0:self.n_components]
        self._singular_values = S[0:self.n_components]
        self._explained_variance_ratio = eigen_values[0:self.n_components] / eigen_values.sum()

    def _transform(self, X: torch.Tensor) -> torch.Tensor:
        return X.matmul(self._weight)

    @property
    def weight(self) -> torch.Tensor:
        return self._weight

    @property
    def singular_values(self) -> torch.Tensor:
        return self._singular_values

    @property
    def explained_variance_ratio(self) -> torch.Tensor:
        return self._explained_variance_ratio