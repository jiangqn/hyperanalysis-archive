import torch
from hyperanalysis.utils.linalg import cov, postive_definite_matrix_power
from typing import Tuple

class CCA(object):

    def __init__(self, n_components=None):
        super(CCA, self).__init__()
        self.n_components_ = n_components

    def fit(self, X: torch.FloatTensor, Y: torch.FloatTensor) -> None:
        self._fit(X, Y)

    def transform(self, X: torch.FloatTensor, Y: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        return self._transform(X, Y)

    def fit_transform(self, X: torch.FloatTensor, Y: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        self._fit(X, Y)
        return self._transform(X, Y)

    def _fit(self, X: torch.FloatTensor, Y: torch.FloatTensor) -> None:

        assert len(X.size()) == 2
        assert len(Y.size()) == 2
        assert X.size(0) == Y.size(0)
        num = X.size(0)
        X_dim = X.size(1)
        Y_dim = Y.size(1)
        assert num >= 2

        if self.n_components_ == None:
            self.n_components_ = min(X_dim, Y_dim)

        assert isinstance(self.n_components_, int)
        assert self.n_components_ >= 1 and self.n_components_ <= min(num, X_dim, Y_dim)

        self.X_mean_ = X.mean(dim=0, keepdim=True)
        self.Y_mean_ = Y.mean(dim=0, keepdim=True)
        X -= self.X_mean_
        Y -= self.Y_mean_

        SXX = cov(X, X)
        SXY = cov(X, Y)
        SYY = cov(Y, Y)

        PSXX = postive_definite_matrix_power(SXX, -1 / 2)
        PSYY = postive_definite_matrix_power(SYY, -1 / 2)

        M = PSXX.matmul(SXY).matmul(PSYY)
        U, S, V = torch.svd(M)

        self.corr_coefs_ = S[0:self.n_components_].tolist()
        self.A_ = PSXX.matmul(U[:, 0:self.n_components_])
        self.B_ = PSYY.matmul(V[:, 0:self.n_components_])

    def _transform(self, X: torch.FloatTensor, Y: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        X = (X - self.X_mean_).matmul(self.A_)
        Y = (Y - self.Y_mean_).matmul(self.B_)
        return X, Y