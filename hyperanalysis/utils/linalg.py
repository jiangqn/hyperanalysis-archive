import torch

def cov(X: torch.FloatTensor, Y: torch.FloatTensor = None) -> torch.FloatTensor:
    """
    :param X: (num, x_dim)
    :param Y: (num, y_dim)
    :return C: (x_dim, y_dim) covariance matrix
    """

    if Y == None:
        Y = X

    assert X.size(0) == Y.size(0)
    num = X.size(0)
    assert num >= 2

    X_mean = X.mean(dim=0, keepdim=True)
    Y_mean = Y.mean(dim=0, keepdim=True)
    X = X - X_mean
    Y = Y - Y_mean
    C = X.t().matmul(Y) / (num - 1)
    return C