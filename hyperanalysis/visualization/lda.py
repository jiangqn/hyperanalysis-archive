import torch
# from hyperanalysis.utils.linalg import postive_definite_matrix_power, cov
from hyperanalysis.utils import linalg

def linear_discriminant_analysis(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    """
    :param X: torch.FloatTensor (num, dim)
    :param y: torch.LongTensor (num,)
    :return Z: torch.FloatTensor (num, 2)
    """

    assert len(X.size()) == 2
    assert len(y.size()) == 1
    assert X.size(0) == y.size(0)

    dtype = X.dtype
    device = X.device

    num, dim = X.size()
    K = y.max().item() + 1

    mean = torch.zeros(K, dim, dtype=dtype, device=device)
    covariance = torch.zeros(K, dim, dim, dtype=dtype, device=device)
    count = torch.zeros(K, dtype=dtype, device=device)
    XC = []

    Sw = torch.zeros(dim, dim, dtype=X.dtype, device=X.device)

    for k in range(K):

        index = torch.arange(num, device=X.device)
        mask = (y == k)
        nk = mask.sum().item()
        count[k] = nk
        index = index.masked_select(mask)
        sX = X.index_select(dim=0, index=index)
        mean[k] = sX.mean(dim=0, keepdim=False)
        covariance[k] = linalg.cov(sX)
        XC.append(sX - mean[k].unsqueeze(0))

    prior = count / num
    xbar = mean.matmul(prior)

    XC = torch.cat(XC, dim=0)
    std = XC.mean(dim=0)
    factor = 1.0 / (num - K)
    X = torch.sqrt(factor) * (XC / std)
    U, s, Vt = torch.svd(X)

        # m[k] = mk[0]
        # Sw += sX.t().matmul(sX) / nk - mk.t().matmul(mk)

    # squared_m_norm = (m * m).sum(dim=0, keepdim=True)
    # Sb = squared_m_norm + squared_m_norm.t() - 2 * m.t().matmul(m)
    # Sb = Sb - Sb.min()
    #
    # pSw = postive_definite_matrix_power(Sw, -1/2)
    # S = pSw.matmul(Sb).matmul(pSw)
    # # S = torch.inverse(Sw).matmul(Sb)
    # eigenvalues, eigenvectors = torch.eig(S, eigenvectors=True)
    # eigenvalues = eigenvalues[:, 0]
    #
    # print(eigenvalues)

    if K == 2:

        _, index = eigenvalues.max(dim=0)
        eigenvector = eigenvectors[index]

        U = eigenvector.unsqueeze(-1)
        U = U / U.norm(dim=0, keepdim=True)
        zu = X.matmul(U)

        T = torch.eye(dim, dtype=X.dtype, device=X.device)
        T[:, 0] = U[:, 0]
        T, _ = torch.qr(T)
        T[:, 0] = U[:, 0]

        tX = X.matmul(T)
        tX = tX[:, 1:]
        cov = tX.t().matmul(tX)

        eigenvalues, eigenvectors = torch.eig(cov, eigenvectors=True)
        eigenvalues = eigenvalues[:, 0]
        # _, index = eigenvalues.max(dim=0)
        eigenvector = eigenvectors[0]

        zero = torch.zeros(1, dtype=X.dtype, device=X.device)
        v = torch.cat([zero, eigenvector], dim=0)
        V = v.unsqueeze(-1)
        V = T.matmul(V)
        V = V / V.norm(dim=0, keepdim=True)
        zv = X.matmul(V)

        Z = torch.cat([zu, zv], dim=-1)

    else: # K > 2

        # sorted_indices = eigenvalues.argsort(dim=0, descending=True)

        # P = eigenvectors[:, sorted_indices[0, 2]]
        #
        P = eigenvectors[:, [0, 2]]
        P = P / P.norm(dim=0, keepdim=True)
        Z = X.matmul(P)

    return Z