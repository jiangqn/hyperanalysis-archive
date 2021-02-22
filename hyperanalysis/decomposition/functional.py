import torch
from hyperanalysis.decomposition.pca import PCA
from hyperanalysis.decomposition.truncated_svd import TruncatedSVD
from hyperanalysis.decomposition.lda import LinearDiscriminantAnalysis

def pca(X: torch.Tensor, n_components: int = 2) -> torch.Tensor:
    """
    :param X: FloatTensor (num, dim)
    :param n_components: int
    :return : FloatTensor (num, n_components)
    """
    model = PCA(n_components=n_components)
    return model.fit_transform(X)

def truncated_svd(X: torch.Tensor, n_components: int = 2) -> torch.Tensor:
    """
    :param X: FloatTensor (num, dim)
    :param n_components: int
    :return : FloatTensor (num, n_components)
    """
    model = TruncatedSVD(n_components=n_components)
    return model.fit_transform(X)

def lda(X: torch.Tensor, y: torch.LongTensor, n_components: int = 1) -> torch.Tensor:
    """
    :param X: FloatTensor (num, dim)
    :param y: FloatTensor (num,)
    :param n_components: int
    :return : FloatTensor (num, n_components)
    """
    model = LinearDiscriminantAnalysis(n_components=n_components)
    return model.fit_transform(X, y)