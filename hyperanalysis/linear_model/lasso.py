import torch
import math

class Lasso(object):

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True, max_iter: int = 1000) -> None:
        super(Lasso, self).__init__()
        assert alpha >= 0
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter

    def fit(self, X: torch.FloatTensor, y: torch.FloatTensor) -> None:
        self._fit(X, y)

    def predict(self, X: torch.FloatTensor) -> torch.FloatTensor:
        return self._predict(X)

    def fit_predict(self, X: torch.FloatTensor, y: torch.FloatTensor) -> torch.FloatTensor:
        self._fit(X, y)
        return self._predict(X)

    def _fit(self, X: torch.FloatTensor, y: torch.FloatTensor) -> None:

        X, y = X.clone(), y.clone()

        assert len(X.size()) == 2
        assert len(y.size()) == 1
        assert X.size(0) == y.size(0)
        num = X.size(0)

        if self.fit_intercept:
            constant = torch.ones((num, 1), dtype=X.dtype, device=X.device)
            X = torch.cat((constant, X), dim=1)
        dim = X.size(1)

        # optimize Lasso with coordinate descent algorithm

        self.beta = torch.randn((dim, 1), dtype=X.dtype, device=X.device)

        for i in range(1, self.max_iter + 1):

            for j in range(dim):
                y_pred = X.matmul(self.beta).squeeze(-1)
                rho = (X[:, j] * (y - y_pred + self.beta[j, 0] * X[:, j])).mean().item()

                if math.isnan(rho):
                    raise ValueError("rho is nan")
                else:
                    print(i, j, rho)

                if self.fit_intercept and j == 0:
                    self.beta[j] = rho
                else:
                    self.beta[j] = float(rho < -self.alpha) * (rho + self.alpha) + float(rho > self.alpha) * (rho - self.alpha)

        beta = self.beta.unsqueeze(-1).tolist()
        if self.fit_intercept:
            self.coef_ = beta[1:]
            self.intercept_ = beta[0]
        else:
            self.coef_ = beta
            self.intercept_ = 0.0

    def _predict(self, X: torch.FloatTensor) -> torch.FloatTensor:

        assert hasattr(self, "coef_")
        assert hasattr(self, "intercept_")

        X = X.clone()

        assert len(X.size()) == 2
        num = X.size(0)

        if self.fit_intercept:
            constant = torch.ones((num, 1), dtype=X.dtype, device=X.device)
            X = torch.cat((constant, X), dim=1)

        y = X.matmul(self.beta).unsqueeze(-1)
        return y