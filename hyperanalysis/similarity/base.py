import torch

class RepresentationSimilarity(object):

    def __init__(self) -> None:
        super(RepresentationSimilarity, self).__init__()

    def validate_inputs(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        pass

    def score(self, X: torch.Tensor, Y: torch.Tensor):
        pass

    def _score(self, X: torch.Tensor, Y: torch.Tensor):
        pass