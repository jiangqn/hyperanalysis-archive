import torch
from hyperanalysis.visualization.entended_lda import ExtendedLDA
from hyperanalysis.visualization.extended_lra import ExtendedLRA

def extended_lda(X: torch.Tensor, y: torch.LongTensor) -> torch.Tensor:
    """
    :param X: FloatTensor (num, dim)
    :param y: LongTensor (num,)
    :return : FloatTensor (num, 2)
    """
    model = ExtendedLDA()
    return model.fit_transform(X, y)

def extended_lra(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    :param X: FloatTensor (num, dim)
    :param y: FloatTensor (num,)
    :return : FloatTensor (num, 2)
    """
    model = ExtendedLRA()
    return model.fit_transform(X, y)

def tsne(X: torch.Tensor) -> torch.Tensor:
    pass