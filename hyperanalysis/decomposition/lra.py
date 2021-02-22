import torch
from hyperanalysis.decomposition.base import SupervisedDecomposition

class LinearRegressionAnalysis(SupervisedDecomposition):

    def __init__(self) -> None:
        super(LinearRegressionAnalysis, self).__init__(n_components=1)

    def _fit(self, X: torch.Tensor, y: torch.Tensor) -> None:

        self._mean = X.mean(dim=0, keepdim=True)
        X = X - self._mean

        num, dim = X.size()
        ones = torch.ones(size=(num, 1), dtype=X.dtype, device=X.device)
        X = torch.cat((X, ones), dim=1)
        Y = y.unsqueeze(-1)

        U = torch.inverse(X.t().matmul(X)).matmul(X.t()).matmul(Y)

        weight = U[0:-1, 0:1]
        self._weight = weight / weight.norm(dim=0)

    def _transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self._mean).matmul(self._weight)

    @property
    def mean(self) -> torch.Tensor:
        """
        :return : FloatTensor (1, dim)
        """
        assert self.is_trained
        return self._mean

    @property
    def weight(self) -> torch.Tensor:
        """
        :return : FloatTensor (dim, 1)
        """
        assert self.is_trained
        return self._weight