import torch

eps = 1e-6

def cov(X: torch.Tensor, Y: torch.Tensor = None) -> torch.Tensor:
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

def unnormal_cov(X: torch.Tensor, Y: torch.Tensor = None) -> torch.Tensor:
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
    C = X.t().matmul(Y)
    return C

def squared_euclidean_distance(X: torch.Tensor, Y: torch.Tensor = None) -> torch.Tensor:
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

def check_matrix_symmetric(X: torch.Tensor) -> None:
    assert len(X.size()) == 2
    assert X.size(0) == X.size(1)
    assert (X - X.t()).abs().max().item() <= eps

def postive_definite_matrix_power(X: torch.Tensor, power=1) -> torch.Tensor:
    check_matrix_symmetric(X)
    U, S, V = torch.svd(X)
    return U.matmul(torch.diag(torch.pow(S, power))).matmul(V.t())

def constrained_max_variance(X: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    :param X: torch.Tensor (num, dim)
    :param u: torch.Tensor (dim,)
    :return v: torch.Tensor (dim,)
    """

    X, u = X.clone(), u.clone()

    assert len(X.size()) == 2
    assert len(u.size()) == 1
    assert X.size(1) == u.size(0)

    X = X - X.mean(dim=0, keepdim=True)
    u = u / u.norm()
    num, dim = X.size()

    T = torch.eye(dim, dtype=X.dtype, device=X.device)
    T[:, 0] = u
    Q, _ = torch.qr(T)
    Q[:, 0] = u

    qX = X.matmul(Q)
    qX = qX[:, 1:]
    cov = qX.t().matmul(qX)

    eigenvalues, eigenvectors = torch.eig(cov, eigenvectors=True)
    eigenvalues = eigenvalues[:, 0]
    _, index = eigenvalues.max(dim=0)
    eigenvector = eigenvectors[index]

    zero = torch.zeros(1, dtype=X.dtype, device=X.device)
    v = torch.cat([zero, eigenvector], dim=0)
    v = Q.matmul(v)
    v = v / v.norm()

    return v

def constrained_max_correlation(X: torch.Tensor, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    :param X: torch.Tensor (num, dim)
    :param y: torch.Tensor (num,)
    :param u: torch.Tensor (dim,)
    :return v: torch.Tensor (dim,)
    """

    X, y, u = X.clone(), y.clone(), u.clone()

    assert len(X.size()) == 2
    assert len(u.size()) == 1
    assert X.size(1) == u.size(0)

    X = X - X.mean(dim=0, keepdim=True)
    u = u / u.norm()
    num, dim = X.size()

    T = torch.eye(dim, dtype=X.dtype, device=X.device)
    T[:, 0] = u
    Q, _ = torch.qr(T)
    Q[:, 0] = u

    qX = X.matmul(Q)
    qX[:, 0] = torch.ones(num, dtype=X.dtype, device=X.device)

    v = torch.inverse(qX.t().matmul(qX)).matmul(qX.t()).matmul(y)
    v[0] = 0
    v = Q.matmul(v)
    v = v / v.norm()

    return v

def multiple_correlation(X: torch.Tensor, y: torch.Tensor) -> float:
    """
    :param X: torch.Tensor (num, dim)
    :param y: torch.Tensor (num,)
    """
    assert len(X.size()) == 2
    assert len(y.size()) == 1
    assert X.size(0) == y.size(0)
    num = X.size(0)
    ones = torch.ones(size=(num, 1), dtype=X.dtype, device=X.device)
    X = torch.cat((X, ones), dim=1)
    Y = y.unsqueeze(-1)
    W = torch.inverse(X.t().matmul(X)).matmul(X.t()).matmul(Y)  # torch.Tensor (dim + 1, 1)
    print(W)
    Z = X.matmul(W)
    z = Z.squeeze(-1)
    y = y - y.mean()
    z = z - z.mean()
    correlation = y.dot(z) / (torch.norm(y) * torch.norm(z))
    return correlation.item()