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

def squared_euclidean_distance(X: torch.FloatTensor, Y: torch.FloatTensor = None) -> torch.FloatTensor:
    """
    :param X: (x_num, dim)
    :param Y: (y_num, dim)
    :return D: (x_num, y_num) squared L2 distance D_{ij} = ||x_i - y_j||_{2}^{2}
    """

    if Y == None:
        Y = X

    G = X.matmul(Y.t())
    squared_X_norm = (X * X).sum(dim=1, keepdim=True)
    squared_Y_norm = (Y * Y).sum(dim=1, keepdim=True).t()
    D = squared_X_norm + squared_Y_norm - 2 * G
    D = D - D.min()
    return D

def check_matrix_symmetric(X: torch.FloatTensor) -> None:
    assert len(X.size()) == 2
    assert X.size(0) == X.size(1)
    assert (X - X.t()).abs().max().item() <= eps

def postive_definite_matrix_power(X: torch.FloatTensor, power=1) -> torch.FloatTensor:
    check_matrix_symmetric(X)
    U, S, V = torch.svd(X)
    return U.matmul(torch.diag(torch.pow(S, power))).matmul(V.t())