import torch

class CCA(object):

    def __init__(self, n_components=None):
        super(CCA, self).__init__()
        self.n_components = n_components

    def _fit(self, X: torch.FloatTensor, Y: torch.FloatTensor) -> None:

        assert len(X.size()) == 2
        assert len(Y.size()) == 2
        assert X.size(0) == Y.size(0)

        if self.n_components == None:
            self.n_components = min(X.size(1), Y.size(1))

    def _transform(self, X: torch.FloatTensor) -> torch.FloatTensor:
        pass