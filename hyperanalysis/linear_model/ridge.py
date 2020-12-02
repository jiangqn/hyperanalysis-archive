import torch

class Ridge(object):

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X: torch.FloatTensor, y: torch.FloatTensor) -> None:
        self._fit(X, y)

    def transform(self, X: torch.FloatTensor) -> torch.FloatTensor:
        return self._transform(X)

    def fit_transform(self, X: torch.FloatTensor, y: torch.FloatTensor) -> torch.FloatTensor:
        self._fit(X, y)
        return self._transform(X)

    def _fit(self, X: torch.FloatTensor, y: torch.FloatTensor) -> torch.FloatTensor:

        X, y = X.clone(), y.clone()

        assert len(X.size()) == 2
        assert len(y.size()) == 1
        assert X.size(0) == y.size(0)
        num = X.size(0)

        if self.fit_intercept:
            constant = torch.ones((num, 1), dtype=X.dtype, device=X.device)
            X = torch.cat((X, constant), dim=1)
        y = y.unsqueeze(-1)

        dim = X.size(1)
        I = torch.eye(dim, dtype=X.dtype, device=X.device)
        self.beta = torch.inverse(X.t().matmul(X) + self.alpha * I).matmul(X.t()).matmul(y)

        beta = self.beta.unsqueeze(-1).tolist()
        if self.fit_intercept:
            self.coef_ = beta[0:-1]
            self.intercept_ = beta[-1]
        else:
            self.coef_ = beta
            self.intercept_ = 0.0

    def _transform(self, X: torch.FloatTensor) -> torch.FloatTensor:

        assert hasattr(self, "coef_")
        assert hasattr(self, "intercept_")

        X = X.clone()

        assert len(X.size()) == 2
        num = X.size(0)

        if self.fit_intercept:
            constant = torch.ones((num, 1), dtype=X.dtype, device=X.device)
            X = torch.cat((X, constant), dim=1)

        y = X.matmul(self.beta).unsqueeze(-1)
        return y