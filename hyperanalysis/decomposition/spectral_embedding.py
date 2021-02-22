import torch
from hyperanalysis.decomposition.base import UnsupervisedDecomposition
from hyperanalysis.kernel import Kernel

class SpectralEmbedding(UnsupervisedDecomposition):

    def __init__(self, n_components: int, kernel: Kernel) -> None:
        super(SpectralEmbedding, self).__init__(n_components)
        self._kernel = Kernel

    def _fit(self, X: torch.Tensor) -> None:
        pass