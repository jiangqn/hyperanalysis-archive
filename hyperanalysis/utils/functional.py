import torch
from hyperanalysis.decomposition.pca import PCA
from hyperanalysis.decomposition.truncated_svd import TruncatedSVD
from hyperanalysis.representation_similarity.svcca import SVCCA

def pca(X: torch.FloatTensor, n_components: int = None, explained_variance_ratio: float = None) -> torch.FloatTensor:
    _pca = PCA(n_components=n_components, explained_variance_ratio=explained_variance_ratio)
    return _pca.fit_transform(X)

def truncated_svd(X: torch.FloatTensor, n_components: int = None, explained_variance_ratio: float = None) -> torch.FloatTensor:
    _truncated_svd = TruncatedSVD(n_components=n_components, explained_variance_ratio=explained_variance_ratio)
    return _truncated_svd.fit_transform(X)

def svcca(X: torch.FloatTensor, Y: torch.FloatTensor, explained_variance_ratio: float = 0.99) -> float:
    _svcca = SVCCA(explained_variance_ratio=explained_variance_ratio)
    return _svcca.score(X, Y)