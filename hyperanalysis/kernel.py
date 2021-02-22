import torch
from hyperanalysis.linalg import squared_euclidean_distance

class Kernel(object):

    def __init__(self) -> None:
        super(Kernel, self).__init__()

    def __call__(self, X: torch.FloatTensor, Y: torch.FloatTensor = None) -> torch.FloatTensor:

        if Y == None:
            Y = X

        assert len(X.size()) == 2 and len(Y.size()) == 2
        assert X.size(1) == Y.size(1)

        return self._compute(X, Y)

    def _compute(self, X: torch.FloatTensor, Y: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError("The _compute method in class Kernel is not implemented.")

class LinearKernel(Kernel):

    def __init__(self) -> None:
        super(LinearKernel, self).__init__()

    def _compute(self, X: torch.FloatTensor, Y: torch.FloatTensor) -> torch.FloatTensor:
        return X.matmul(Y.t())

class PolynomialKernel(Kernel):

    def __init__(self, degree: int = 2, coef0 : float = 1.0) -> None:
        super(PolynomialKernel, self).__init__()
        self.degree = degree
        self.coef0 = coef0

    def _compute(self, X: torch.FloatTensor, Y: torch.FloatTensor) -> torch.FloatTensor:
        K = torch.pow(X.matmul(Y.t()) + self.coef0, self.degree)
        return K

class RBFKernel(Kernel):

    def __init__(self, sigma: float = None) -> None:
        super(RBFKernel, self).__init__()
        self.sigma = sigma

    def _compute(self, X: torch.FloatTensor, Y: torch.FloatTensor) -> torch.FloatTensor:
        S = squared_euclidean_distance(X, Y)
        if self.sigma == None:
            self.sigma = torch.sqrt(torch.median(S[S != 0])).item()
        K = torch.exp(S / (-2 * self.sigma * self.sigma))
        return K

def get_kernel(kernel: str = "linear", sigma: float = None, degree: int = 2, coef0: float = 1.0) -> Kernel:

    assert kernel in ["linear", "poly", "rbf"]

    if kernel == "linear":
        _kernel = LinearKernel()

    elif kernel == "poly":
        _kernel = PolynomialKernel(degree=degree, coef0=coef0)

    else:   # kernel == "rbf"
        _kernel = RBFKernel(sigma=sigma)

    return _kernel