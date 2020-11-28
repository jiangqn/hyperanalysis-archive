import torch

class PCA(object):

    def __init__(self, n_components=None):
        super(PCA, self).__init__()
        self.n_components = n_components

    def fit(self, X: torch.FloatTensor) -> None:
        self._fit(X)

    def transform(self, X: torch.FloatTensor) -> torch.FloatTensor:
        return self._transform(X)

    def fit_transform(self, X: torch.FloatTensor) -> torch.FloatTensor:
        self._fit(X)
        return self._transform(X)

    def _fit(self, X: torch.FloatTensor) -> None:

        assert len(X.size()) == 2
        num, dim = X.size()

        if self.n_components == None:
            self.n_components = min(num, dim)

        _, S, V = torch.svd(X)

        index = S.argsort(descending=True)
        S = S.index_select(index=index, dim=0)
        V = V.index_select(index=index, dim=1)

        eigenvalues = S * S / (num - 1)
        self.explained_variance_ = eigenvalues[0:self.n_components].tolist()
        self.explained_variance_ratio_ = (eigenvalues[0:self.n_components] / eigenvalues.sum()).tolist()
        self.V_ = V[:, 0:self.n_components]

    def _transform(self, X: torch.FloatTensor) -> torch.FloatTensor:
        assert hasattr(self, "V_")
        return X.matmul(self.V_)