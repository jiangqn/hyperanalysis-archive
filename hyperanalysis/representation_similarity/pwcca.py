import torch
from hyperanalysis.decomposition.truncated_svd import TruncatedSVD
from hyperanalysis.cross_decomposition.cca import CCA

class PWCCA(object):

    def __init__(self, explained_variance_ratio: float = 0.99, symmetric: bool = False) -> None:
        super(PWCCA, self).__init__()
        self._explained_variance_ratio = explained_variance_ratio
        self.symmetric = symmetric

    def score(self, X: torch.FloatTensor, Y: torch.FloatTensor) -> float:

        X, Y = X.clone(), Y.clone()

        X_tsvd = TruncatedSVD(explained_variance_ratio=self._explained_variance_ratio)
        X = X_tsvd.fit_transform(X)
        Y_tsvd = TruncatedSVD(explained_variance_ratio=self._explained_variance_ratio)
        Y = Y_tsvd.fit_transform(Y)

        X_dim = X.size(1)
        Y_dim = Y.size(1)
        self.n_components_ = min(X_dim, Y_dim)

        cca = CCA(n_components=self.n_components_)
        cca.fit(X, Y)

        self.corr_coefs_ = cca.corr_coefs_
        assert len(self.corr_coefs_) == self.n_components_

        X_weights = X.matmul(cca.A_).abs().sum(dim=0)
        X_weights = X_weights / X_weights.sum()

        Y_weights = Y.matmul(cca.B_).abs().sum(dim=0)
        Y_weights = Y_weights / Y_weights.sum()

        if self.symmetric:
            weights = (X_weights + Y_weights) / 2
        else:
            if X_dim <= Y_dim:
                weights = X_weights
            else:
                weights = Y_weights
        self.weights_ = weights.tolist()

        assert len(self.weights_) == self.n_components_

        self.pwcca_score_ = sum(c * w for c, w in zip(self.corr_coefs_, self.weights_))
        return self.pwcca_score_