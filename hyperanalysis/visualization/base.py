import torch
from hyperanalysis.decomposition.base import UnsupervisedDecomposition, SupervisedDecomposition

class UnsupervisedVisualization(UnsupervisedDecomposition):

    def __init__(self) -> None:
        super(UnsupervisedVisualization, self).__init__(n_components=2)

class SupervisedVisualization(SupervisedDecomposition):

    def __init__(self) -> None:
        super(SupervisedVisualization, self).__init__(n_components=2)