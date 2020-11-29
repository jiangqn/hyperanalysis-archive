import torch
from hyperanalysis.decomposition.pca import PCA
from hyperanalysis.decomposition.truncated_svd import TruncatedSVD

def pca(X: torch.FloatTensor, n_components: int = None, explained_variance_ratio: float = None) -> torch.FloatTensor:
    _pca = PCA(n_components=n_components, explained_variance_ratio=explained_variance_ratio)
    return _pca.fit_transform(X)

def truncated_svd(X: torch.FloatTensor, n_components: int = None, explained_variance_ratio: float = None) -> torch.FloatTensor:
    _truncated_svd = TruncatedSVD(n_components=n_components, explained_variance_ratio=explained_variance_ratio)
    return _truncated_svd.fit_transform(X)