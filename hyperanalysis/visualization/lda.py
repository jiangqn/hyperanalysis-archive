import torch

def linear_discriminant_analysis(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    """
    :param X: torch.FloatTensor (num, dim)
    :param y: torch.LongTensor (num,)
    :return Z: torch.FloatTensor (num, 2)
    """

    assert len(X.size()) == 2
    assert len(y.size()) == 1
    assert X.size(0) == y.size(0)

    num, dim = X.size()
    K = y.max().item() + 1
    m = torch.zeros(K, dim, dtype=X.dtype, device=X.device)

    Sw = torch.zeros(dim, dim, dtype=X.dtype, device=X.device)

    for k in range(K):
        index = torch.arange(num, device=X.device)
        mask = (y == k)
        nk = mask.sum().item()
        index = index.masked_select(mask)
        sX = X.index_select(dim=0, index=index)
        mk = sX.mean(dim=0, keepdim=True)
        m[k] = mk[0]
        Sw += sX.t().matmul(sX) / nk - mk.t().matmul(mk)

    squared_m_norm = (m * m).sum(dim=0, keepdim=True)
    Sb = squared_m_norm + squared_m_norm.t() - 2 * m.t().matmul(m)
    Sb = Sb - Sb.min()

    S = torch.inverse(Sw).matmul(Sb)
    eigenvalues, eigenvectors = torch.eig(S, eigenvectors=True)
    eigenvalues = eigenvalues[:, 0]
    _, index = eigenvalues.max(dim=0)
    eigenvector = eigenvectors[index]

    U = eigenvector.unsqueeze(-1)
    U = U / U.norm(dim=0, keepdim=True)
    zu = X.matmul(U)

    X = X - X.mean(dim=0, keepdim=True)

    T = torch.randn(dim, dim, dtype=X.dtype, device=X.device)
    T[:, 0] = U[:, 0]
    T, _ = torch.qr(T)
    T[:, 0] = U[:, 0]

    tX = X.matmul(T)
    tX = tX[:, 1:]
    cov = tX.t().matmul(tX)

    eigenvalues, eigenvectors = torch.eig(cov, eigenvectors=True)
    eigenvalues = eigenvalues[:, 0]
    _, index = eigenvalues.max(dim=0)
    eigenvector = eigenvectors[index]

    zero = torch.zeros(1, dtype=X.dtype, device=X.device)
    v = torch.cat([zero, eigenvector], dim=0)
    V = v.unsqueeze(-1)
    V = T.matmul(V)
    V = V / V.norm(dim=0, keepdim=True)
    zv = X.matmul(V)

    Z = torch.cat([zu, zv], dim=-1)
    return Z