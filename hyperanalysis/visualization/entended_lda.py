import torch
from hyperanalysis import linalg
from hyperanalysis.visualization.base import SupervisedVisualization

class ExtendedLDA(SupervisedVisualization):

    def __init__(self) -> None:
        super(ExtendedLDA, self).__init__()

    def _fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        :param X: FloatTensor (num, dim)
        :param y: LongTensor (num,)
        """

        dtype = X.dtype
        device = X.device

        self._mean = X.mean(dim=0, keepdim=True)

        X = X - self._mean

        num, dim = X.size()
        K = y.max().item() + 1

        St = linalg.unnormal_cov(X)
        Sw = torch.zeros(dim, dim, dtype=dtype, device=device)

        for k in range(K):
            index = torch.arange(num, device=X.device)
            mask = (y == k)
            # nk = mask.sum().item()
            index = index.masked_select(mask)
            Xk = X.index_select(dim=0, index=index)
            Sw = Sw + linalg.unnormal_cov(Xk)

        Sb = St - Sw

        Sb = Sb / num
        Sw = Sw / num

        if K == 2:

            pSw = linalg.postive_definite_matrix_power(Sw, -0.5)
            pSb = linalg.postive_definite_matrix_power(Sb, 0.5)

            B = pSb.matmul(pSw)
            _, s, V = torch.svd(B)

            sorted_indices = s.argsort(dim=0, descending=True)
            p = V[:, sorted_indices[0]]
            u = pSw.matmul(p)
            u = u / u.norm()
            v = linalg.constrained_max_variance(X, u)

            W = torch.stack([u, v], dim=1)
            self._weight = W

        else:  # K > 2

            pSw = linalg.postive_definite_matrix_power(Sw, -0.5)
            pSb = linalg.postive_definite_matrix_power(Sb, 0.5)

            B = pSb.matmul(pSw)
            _, s, V = torch.svd(B)

            sorted_indices = s.argsort(dim=0, descending=True)
            P = V[:, sorted_indices[0:2]]
            W = pSw.matmul(P)
            self._weight = W / W.norm(dim=0, keepdim=True)

    def _transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: FloatTensor (num, dim)
        :param y: LongTensor (num,)
        :return : FloatTensor (num, 2)
        """
        return (X - self._mean).matmul(self._weight)

    @property
    def mean(self) -> torch.Tensor:
        assert self.is_trained
        return self._mean

    @property
    def weight(self) -> torch.Tensor:
        assert self.is_trained
        return self._weight