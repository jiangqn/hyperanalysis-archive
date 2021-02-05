import torch
from hyperanalysis.decomposition.pca import PCA
from hyperanalysis.decomposition.truncated_svd import TruncatedSVD
from hyperanalysis.similarity.svcca import SVCCA
from hyperanalysis.similarity.pwcca import PWCCA
from hyperanalysis.similarity.cka import CKA

def pca(X: torch.FloatTensor, n_components: int = None, explained_variance_ratio: float = None) -> torch.FloatTensor:
    _pca = PCA(n_components=n_components, explained_variance_ratio=explained_variance_ratio)
    return _pca.fit_transform(X)

def truncated_svd(X: torch.FloatTensor, n_components: int = None, explained_variance_ratio: float = None) -> torch.FloatTensor:
    _truncated_svd = TruncatedSVD(n_components=n_components, explained_variance_ratio=explained_variance_ratio)
    return _truncated_svd.fit_transform(X)

def svcca(X: torch.FloatTensor, Y: torch.FloatTensor, explained_variance_ratio: float = 0.99) -> float:
    _svcca = SVCCA(explained_variance_ratio=explained_variance_ratio)
    return _svcca.score(X, Y)

def pwcca(X: torch.FloatTensor, Y: torch.FloatTensor, explained_variance_ratio: float = 0.99, symmetric: bool = False) -> float:
    _pwcca = PWCCA(explained_variance_ratio=explained_variance_ratio, symmetric=symmetric)
    return _pwcca.score(X, Y)

def cka(X: torch.FloatTensor, Y: torch.FloatTensor, kernel: str = "linear", sigma: float = None, degree: int = None) -> float:
    _cka = CKA(kernel=kernel, sigma=sigma, degree=degree)
    return _cka.score(X, Y)