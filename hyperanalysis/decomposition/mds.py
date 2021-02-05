import torch

class MDS(object):

    def __init__(self, n_components=None) -> None:
        super(MDS, self).__init__()
        self.n_components = n_components

    def fit(self, X: torch.FloatTensor) -> None:
        pass

    def fit_transform(self, X: torch.FloatTensor) -> torch.FloatTensor:
        pass

    def transform(self, X: torch.FloatTensor) -> torch.FloatTensor:
        pass

    def _fit(self, X: torch.FloatTensor) -> None:
        pass

    def _transform(self, X: torch.FloatTensor) -> torch.FloatTensor:
        pass