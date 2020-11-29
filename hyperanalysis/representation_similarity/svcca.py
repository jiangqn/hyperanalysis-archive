import torch
import hyperanalysis.utils.functional as F
from hyperanalysis.cross_decomposition.cca import CCA

class SVCCA(object):

    def __init__(self, explained_variance_ratio: float = 0.99) -> None:
        self._explained_variance_ratio = explained_variance_ratio

    def score(self, X: torch.FloatTensor, Y: torch.FloatTensor) -> None:

        X, Y = X.clone(), Y.clone()

        X = F.truncated_svd(X, explained_variance_ratio=self._explained_variance_ratio)
        Y = F.truncated_svd(Y, explained_variance_ratio=self._explained_variance_ratio)

        X_dim = X.size(1)
        Y_dim = Y.size(1)
        self.n_components_ = min(X_dim, Y_dim)

        cca = CCA(n_components=self.n_components_)
        cca.fit(X, Y)

        self.corr_coefs_ = cca.corr_coefs_
        assert len(self.corr_coefs_) == self.n_components_
        self.svcca_score_ = sum(self.corr_coefs_) / self.n_components_

        return self.svcca_score_