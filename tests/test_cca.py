import sys
sys.path.append("..")
import torch
import numpy as np
from hyperanalysis.cross_decomposition.cca import CCA
from sklearn.cross_decomposition.cca_ import CCA as sCCA
from scipy.stats import pearsonr

path = "../embedding.npy"

X = np.load(path)
X = torch.from_numpy(X).float()

Y = X[:, 100:250]
X = X[:, 0:100]

scca = sCCA(n_components=2, scale=True)
sx, sy = scca.fit_transform(X, Y)

cca = CCA(n_components=2)
x, y = cca.fit_transform(X, Y)
x, y = x.numpy(), y.numpy()

print(pearsonr(sx[:, 0], sy[:, 0]))
print(pearsonr(sx[:, 1], sy[:, 1]))

print(pearsonr(x[:, 0], y[:, 0]))
print(pearsonr(x[:, 1], y[:, 1]))
print(cca.corr_coefs_)

print("sx mean:", sx.mean(axis=0))
print("x mean:", x.mean(axis=0))
print("sx std:", sx.std(axis=0, ddof=1))
print("x std:", x.std(axis=0, ddof=1))

print(sx)
print(x)
print(sy)
print(y)