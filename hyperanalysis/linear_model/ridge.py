import torch
from hyperanalysis.linear_model.base import LinearRegressor

class Ridge(LinearRegressor):

    def __init__(self, fit_bias: bool = True, l2_reg: float = 0.001) -> None:
        super(Ridge, self).__init__()
        assert l2_reg > 0, "l2_reg must be large than 0 in ridge regression."
        self.fit_bias = fit_bias
        self.l2_reg = l2_reg

    def _fit(self, X: torch.Tensor, y: torch.Tensor) -> None:

        num, dim = X.size()

        if self.fit_bias:
            constant = torch.ones((num, 1), dtype=X.dtype, device=X.device)
            X = torch.cat((constant, X), dim=1)
        Y = y.unsqueeze(-1)

        dim = X.size(1)
        I = torch.eye(dim, dtype=X.dtype, device=X.device)
        W = torch.inverse(X.t().matmul(X) + self.l2_reg * I).matmul(X.t()).matmul(Y).unsqueeze(-1)

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