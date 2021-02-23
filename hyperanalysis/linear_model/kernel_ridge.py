import torch
from hyperanalysis.linear_model.base import LinearRegressor
from hyperanalysis.kernel import Kernel, linear_kernel

class KernelRidge(LinearRegressor):

    def __init__(self, kernel: Kernel = linear_kernel) -> None:
        super(KernelRidge, self).__init__()
        self.kernel = kernel

    def _fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        pass

    def _predict(self, X: torch.Tensor) -> torch.Tensor:
        pass