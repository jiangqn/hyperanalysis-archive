import torch
from hyperanalysis.linear_model.base import LinearClassifier

class LogisticRegression(LinearClassifier):

    def __init__(self) -> None:
        super(LogisticRegression, self).__init__()

    def _fit(self, X: torch.Tensor, y: torch.LongTensor) -> None:
        pass

    def _predict(self, X: torch.Tensor) -> torch.LongTensor:
        pass
