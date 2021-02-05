import torch
from hyperanalysis.utils.kernel import get_kernel

class CKA(object):

    def __init__(self, kernel: str = "linear", sigma: float = None, degree: int = 2, coef0: float = 1.0) -> None:
        super(CKA, self).__init__()

        assert kernel in ["linear", "rbf", "poly"]
        self.kernel = get_kernel(kernel=kernel, sigma=sigma, degree=degree, coef0=coef0)

    def score(self, X: torch.FloatTensor, Y: torch.FloatTensor) -> float:

        X, Y = X.clone(), Y.clone()

        assert X.size() == Y.size()
        assert X.size(0) >= 2

        cka_score = self._HSIC(X, Y) / torch.sqrt(self._HSIC(X, X) * self._HSIC(Y, Y))

        self.cka_score_ = cka_score.item()
        return self.cka_score_

    def _centering(self, K: torch.FloatTensor) -> torch.FloatTensor:

        assert len(K.size()) == 2
        assert K.size(0) == K.size(1)
        num = K.size(0)

        H = torch.eye(num, dtype=K.dtype, device=K.device) - torch.ones(num, num, dtype=K.dtype, device=K.device) / num

        return K.matmul(H)

    def _HSIC(self, X: torch.FloatTensor, Y: torch.FloatTensor) -> torch.FloatTensor:

        num = X.size(0)

        XH = self._centering(self.kernel(X))
        YH = XH if X is Y else self._centering(self.kernel(Y))

        return torch.trace(XH.matmul(YH)) / ((num - 1) * (num - 1))