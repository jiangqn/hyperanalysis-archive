import torch
from hyperanalysis.visualization.base import UnsupervisedVisualization

class TSNE(UnsupervisedVisualization):

    def __init__(self) -> None:
        super(TSNE, self).__init__()

    def _fit(self, X: torch.Tensor) -> None:
        pass

    def _transform(self, X: torch.Tensor) -> torch.Tensor:
        pass