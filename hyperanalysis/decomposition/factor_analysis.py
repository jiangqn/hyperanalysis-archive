import torch
from hyperanalysis.decomposition.base import UnsupervisedDecomposition

class FactorAnalysis(UnsupervisedDecomposition):

    def __init__(self, n_components: int = 2) -> None:
        super(FactorAnalysis, self).__init__(n_components)

    