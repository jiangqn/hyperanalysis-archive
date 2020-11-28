import torch

eps = 1e-6

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
    X -= X_mean
    Y -= Y_mean
    C = X.t().matmul(Y) / (num - 1)
    return C

def check_matrix_symmetric(X: torch.FloatTensor) -> None:
    assert len(X.size()) == 2
    assert X.size(0) == X.size(1)
    assert (X - X.t()).abs().max().item() <= eps

def postive_define_matrix_power(X: torch.FloatTensor, power=1) -> torch.FloatTensor:
    check_matrix_symmetric(X)
    U, S, V = torch.svd(X)
    return U.matmul(torch.diag(torch.pow(S, power))).matmul(V.t())