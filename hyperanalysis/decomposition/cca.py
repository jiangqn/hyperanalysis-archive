import torch
from hyperanalysis.linalg import cov, postive_definite_matrix_power
from hyperanalysis.decomposition.base import CrossDecomposition
from typing import Tuple

class CCA(CrossDecomposition):

    def __init__(self, n_components: int = 1) -> None:
        super(CCA, self).__init__(n_components)

    def _fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:

        self._X_mean = X.mean(dim=0, keepdim=True)
        self._Y_mean = Y.mean(dim=0, keepdim=True)

        X_std = X.std(dim=0, keepdim=True)
        X_std[X_std == 0.0] = 1.0
        Y_std = Y.std(dim=0, keepdim=True)
        Y_std[Y_std == 0.0] = 1.0
        self._X_std, self._Y_std = X_std, Y_std

        X = (X - self._X_mean) / self._X_std
        Y = (Y - self._Y_mean) / self._Y_std

        SXX = cov(X, X)
        SXY = cov(X, Y)
        SYY = cov(Y, Y)

        PSXX = postive_definite_matrix_power(SXX, -1 / 2)
        PSYY = postive_definite_matrix_power(SYY, -1 / 2)

        M = PSXX.matmul(SXY).matmul(PSYY)
        U, S, V = torch.svd(M)

        self.corr_coefs_ = S[0:self._n_components]

        A = PSXX.matmul(U[:, 0:self._n_components])
        B = PSYY.matmul(V[:, 0:self._n_components])
        A = A / torch.norm(A, dim=0, keepdim=True)
        B = B / torch.norm(B, dim=0, keepdim=True)

        self._A, self._B = A, B

    def _transform(self, X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        X = (X - self._X_mean) / self._X_std
        Y = (Y - self._Y_mean) / self._Y_std

        X = X.matmul(self._A)
        Y = Y.matmul(self._B)

        return X, Y

    @property
    def X_mean(self) -> torch.Tensor:
        assert self._is_trained
        return self._X_mean

    @property
    def Y_mean(self) -> torch.Tensor:
        assert self._is_trained
        return self._Y_mean