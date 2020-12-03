import torch

class Ridge(object):

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        super(Ridge, self).__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept

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
        y = y.unsqueeze(-1)

        dim = X.size(1)
        I = torch.eye(dim, dtype=X.dtype, device=X.device)
        self.beta = torch.inverse(X.t().matmul(X) + self.alpha * I).matmul(X.t()).matmul(y)

        beta = self.beta.squeeze(-1).tolist()
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

        y = X.matmul(self.beta).squeeze(-1)
        return y