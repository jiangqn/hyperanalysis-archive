import torch
from hyperanalysis.utils.linalg import cov, postive_definite_matrix_power

class Gaussian(object):

    def __init__(self, covariance_type: str = "full") -> None:
        super(Gaussian, self).__init__()
        assert covariance_type in ["full", "diag", "spherical"]
        self.covariance_type = covariance_type

    def fit(self, X: torch.FloatTensor) -> None:
        self._fit(X)

    def _fit(self, X: torch.FloatTensor) -> None:
        self.mean_ = X.mean(dim=0)
        if self.covariance_type == "full":
            self.covariance_ = cov(X)
            self.inverse_covariance_ = postive_definite_matrix_power(self.covariance_, -1)
            sqrt_det = torch.sqrt(torch.det(self.covariance_)).float()
        elif self.covariance_type == "diag":
            covariance_ = X.var(dim=0, keepdim=False)
            self.covariance_ = torch.diag(covariance_)
            self.inverse_covariance_ = torch.diag(1.0 / covariance_)
            sqrt_det = 1.0
            for variance in covariance_.tolist():
                sqrt_det = sqrt_det * variance
        else:   # self.covariance_type == "spherical"
            covariance_ = X.var(dim=0, keepdim=False).mean()
            dim = X.size(1)
            self.covariance_ = torch.eye(dim, dtype=X.dtype, device=X.device) * covariance_
            self.inverse_covariance_ = torch.eye(dim, dtype=X.dtype, device=X.device) * (1.0 / covariance_)
            

    def _predict_proba(self, X: torch.FloatTensor) -> torch.FloatTensor:
        pass