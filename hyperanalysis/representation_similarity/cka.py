import torch

class CKA(object):

    def __init__(self, kernel: str = "linear", sigma: float = None, degree: int = None) -> None:
        super(CKA, self).__init__()

        assert kernel in ["linear", "rbf", "poly"]
        self.kernel = kernel
        self.sigma = sigma
        self.degree = degree

    def score(self, X: torch.FloatTensor, Y: torch.FloatTensor) -> float:

        X, Y = X.clone(), Y.clone()

        assert X.size() == Y.size()
        assert X.size(0) >= 2

        cka_score = self._HSIC(X, Y) / torch.sqrt(self._HSIC(X, X) * self._HSIC(Y, Y))

        self.cka_score_ = cka_score.item()
        return self.cka_score_

    def _kernel_matrix(self, X: torch.FloatTensor) -> torch.FloatTensor:

        if self.kernel == "linear":
            K = X.matmul(X.t())

        elif self.kernel == "rbf":
            G = X.matmul(X.t()) # gram matrix
            S = (torch.diag(G) - G) + (torch.diag(G) - G).t() # square error matrix ||x_i - x_j||_{2}^{2}
            if self.sigma == None:
                self.sigma = torch.sqrt(torch.median(S[S != 0])).item()
            K = torch.exp(S / (-2 * self.sigma * self.sigma))

        else: # kernel == "poly"
            if self.degree == None:
                self.degree = 2
            K = torch.pow(X.matmul(X.t()) + 1, self.degree)

        return K

    def _centering(self, K: torch.FloatTensor) -> torch.FloatTensor:

        assert len(K.size()) == 2
        assert K.size(0) == K.size(1)
        num = K.size(0)

        H = torch.eye(num) - torch.ones(num, num) / num
        H = H.to(K.device)

        return K.matmul(H)

    def _HSIC(self, X: torch.FloatTensor, Y: torch.FloatTensor) -> torch.FloatTensor:

        num = X.size(0)

        XH = self._centering(self._kernel_matrix(X))
        YH = self._centering(self._kernel_matrix(Y))

        return torch.trace(XH.matmul(YH)) / ((num - 1) * (num - 1))