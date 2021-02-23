import torch
from hyperanalysis.linear_model.base import LinearRegressor

class LinearRegression(LinearRegressor):

    def __init__(self, fit_bias: bool = True) -> None:
        super(LinearRegression, self).__init__()
        self.fit_bias = fit_bias

    def _fit(self, X: torch.Tensor, y: torch.Tensor) -> None:

        assert X.size(0) >= X.size(1), "There is no unbiased solution."

        num, dim = X.size()

        if self.fit_bias:
            constant = torch.ones((num, 1), dtype=X.dtype, device=X.device)
            X = torch.cat((constant, X), dim=1)
        Y = y.unsqueeze(-1)

        W = torch.inverse(X.t().matmul(X)).matmul(X.t()).matmul(Y).unsqueeze(-1)

        if self.fit_bias:
            self._weight = W[1:]
            self._bias = W[0]
        else:
            self._weight = W

    def _predict(self, X: torch.Tensor) -> torch.Tensor:
        y = X.matmul(self._weight)
        if self.fit_bias:
            y = y + self._bias
        return y

    @property
    def weight(self) -> torch.Tensor:
        assert self.is_trained
        return self._weight

    @property
    def bias(self) -> torch.Tensor:
        assert self.is_trained
        assert self.fit_bias
        return self._bias

    def r2_score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        self._validate_inputs(X, y)
        self._fit(X, y)
        y_pred = self._predict(X)
        y_mean = y.mean()
        SST = torch.pow(y - y_mean, 2).sum().item()
        SSE = torch.pow(y - y_pred, 2).sum().item()
        r2_score = 1.0 - SSE / SST
        return r2_score