import torch
from hyperanalysis import linalg
from hyperanalysis.visualization.base import SupervisedVisualization

class ExtendedLRA(SupervisedVisualization):

    def __init__(self) -> None:
        super(ExtendedLRA, self).__init__()

    def _fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self._mean = X.mean(dim=0, keepdim=True)
        X = X - self._mean

        num, dim = X.size()
        ones = torch.ones(size=(num, 1), dtype=X.dtype, device=X.device)
        X = torch.cat((X, ones), dim=1)
        Y = y.unsqueeze(-1)

        U = torch.inverse(X.t().matmul(X)).matmul(X.t()).matmul(Y)

        X = X[:, 0:-1]
        u = U[0:-1, 0]
        u = u / u.norm()
        v = linalg.constrained_max_variance(X, u)
        self._weight = torch.stack([u, v], dim=1)

    def _transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self._mean).matmul(self._weight)

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def weight(self) -> torch.Tensor:
        return self._weight