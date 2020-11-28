import sys
sys.path.append("..")
import torch
import numpy as np
from hyperanalysis.cross_decomposition.cca import CCA

path = "../embedding.npy"

X = np.load(path)
X = torch.from_numpy(X).float()

Y = X[:, 100:250]
X = X[:, 0:100]

cca = CCA(n_components=2)
cca.fit(X, Y)
print(cca.corr_)