import torch

class TruncatedSVD(object):

    def __init__(self, n_components=None, explained_variance_ratio=None) -> None:
        super(TruncatedSVD, self).__init__()
        assert n_components == None or explained_variance_ratio == None
        assert explained_variance_ratio == None or (explained_variance_ratio >= 0 and explained_variance_ratio <= 1)
        self.n_components = n_components
        self._explained_variance_ratio = explained_variance_ratio

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
        assert num >= 2

        _, S, V = torch.svd(X)

        index = S.argsort(descending=True)
        S = S.index_select(index=index, dim=0)
        V = V.index_select(index=index, dim=1)

        eigen_values = S * S / (num - 1)

        if self.n_components == None:
            if self._explained_variance_ratio == None:
                self.n_components = min(num, dim)
            else:
                for n_components in range(1, min(num, dim) + 1):
                    explained_variance_ratio = (eigen_values[0:n_components].sum() / eigen_values.sum()).item()
                    if explained_variance_ratio >= self._explained_variance_ratio:
                        self.n_components = n_components
                        break

        assert isinstance(self.n_components, int)
        assert self.n_components >= 1 and self.n_components <= min(num, dim)

        self.singular_values_ = S[0:self.n_components]
        self.explained_variance_ = eigen_values[0:self.n_components].tolist()
        self.explained_variance_ratio_ = (eigen_values[0:self.n_components] / eigen_values.sum()).tolist()
        self.V_ = V[:, 0:self.n_components]

    def _transform(self, X: torch.FloatTensor) -> torch.FloatTensor:
        assert hasattr(self, "V_")
        return X.matmul(self.V_)