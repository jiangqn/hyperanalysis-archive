import torch
from hyperanalysis.visualization.extended_lra import ExtendedLRA

def extended_lda(X: torch.Tensor, y: torch.LongTensor) -> torch.Tensor:
    pass

def extended_lra(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    :param X: FloatTensor (num, dim)
    :param y: FloatTensor (num, dim)
    :return : FloatTensor (num, 2)
    """
    model = ExtendedLRA()
    return model.fit_transform(X, y)

def tsne(X: torch.Tensor) -> torch.Tensor:
    pass