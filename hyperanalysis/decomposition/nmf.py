import torch
from torch import nn
from torch import optim
from typing import Tuple
from hyperanalysis.decomposition.base import UnsupervisedDecomposition
from hyperanalysis.utils.set_seed import set_seed

class NMF(UnsupervisedDecomposition):

    """
    non-negative matrix factorization
    """

    def __init__(self, n_components: int = 2, init: str = "random", solver: str = "mu",
                 tol: float = 1e-6, max_iter: int = 1000, eps: float = 1e-6, verbose: int = 0, random_state: int = 0) -> None:
        super(NMF, self).__init__(n_components)
        self.init = init
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        set_seed(random_state)

    def factorize(self, X: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        self._fit(X)
        return self.W, self.H

    def _init(self, X: torch.FloatTensor) -> None:

        assert self.init in ["random"]
        assert len(X.size()) == 2
        assert (X < 0).long().sum().item() == 0, "There are negative elements in matrix X."

        scale = torch.sqrt(X.mean() / self.n_components).item()

        self.W = torch.rand(size=(X.size(0), self.n_components), dtype=X.dtype, device=X.device) * scale
        self.H = torch.rand(size=(self.n_components, X.size(1)), dtype=X.dtype, device=X.device) * scale

    def _multiplicative_update_solve(self, X: torch.FloatTensor) -> None:

        """
        solve non-negative matrix factorization by "Multiplicative Update solver"
        Lee, D. D., & Seung, S. (2001). Algorithms for non-negative matrix factorization.
        """

        converge = False
        last_mse = None
        for i in range(1, self.max_iter + 1):

            self.W[self.W < self.eps] = self.eps
            self.H[self.H < self.eps] = self.eps
            self.W = self.W * (X.matmul(self.H.t()) / (self.W.matmul(self.H).matmul(self.H.t())))
            self.H = self.H * (self.W.t().matmul(X) / (self.W.t().matmul(self.W).matmul(self.H)))

            mse = torch.pow(X - self.W.matmul(self.H), 2).mean().item()
            if last_mse == None or last_mse - mse >= self.tol:
                if self.verbose == 1:
                    print("[step %d] [mse %.4f]" % (i, mse))
                last_mse = mse
            else:
                converge = True
                break

        if self.verbose == 1:
            print("Finish.")
            if not converge:
                print("Warning! Not Converge.")

        self.mse = mse

    def _gradient_descent_solve(self, X: torch.FloatTensor) -> None:

        """
        solve non-negative matrix factorization by gradient descent with Adam optimizer.

        Kingma, Diederik P., and Jimmy Lei Ba. “Adam: A Method for Stochastic Optimization.” ICLR 2015.
        """

        W = nn.Parameter(self.W)
        H = nn.Parameter(self.H)
        parameters = nn.ParameterList([W, H])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(parameters, lr=0.001)

        converge = False
        last_mse = None
        for i in range(1, self.max_iter + 1):

            optimizer.zero_grad()
            loss = criterion(X, W.matmul(H))
            loss.backward()
            optimizer.step()

            mse = torch.pow(X - self.W.matmul(self.H), 2).mean().item()
            if last_mse == None or last_mse - mse >= self.tol:
                if self.verbose == 1:
                    print("[step %d] [mse %.4f]" % (i, mse))
                last_mse = mse
            else:
                converge = True
                break

        if self.verbose == 1:
            print("Finish.")
            if not converge:
                print("Warning! Not Converge.")

        self.mse = mse

    def _fit(self, X: torch.FloatTensor) -> None:

        self._init(X)

        assert self.solver in ["mu", "gradient"]

        if self.solver == "mu":
            self._multiplicative_update_solve(X)
        else:   # self.solver == "gradient"
            self._gradient_descent_solve(X)