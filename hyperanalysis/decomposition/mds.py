import torch
from hyperanalysis.decomposition.base import UnsupervisedDecomposition

class MDS(UnsupervisedDecomposition):

    def __init__(self, n_components: int = 2) -> None:
        super(MDS, self).__init__(n_components)

    def _fit(self, X: torch.Tensor) -> None:
        pass

    def _transform(self, X: torch.Tensor) -> torch.Tensor:
        pass