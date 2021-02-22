import sys
sys.path.append("..")
from hyperanalysis.decomposition.pca import PCA

import torch
import numpy as np
import os

from sklearn.decomposition.pca import PCA as sPCA
from hyperanalysis.decomposition.functional import pca
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

path = "../embedding.npy"
X = np.load(path)
X = X - X.mean(axis=0)

X = torch.from_numpy(X).float()
print(pca(X).size())

# spca = sPCA(n_components=2)
#
# start = time.time()
# sy = spca.fit_transform(X)
# end = time.time()
# print(end - start)
#
# X = torch.from_numpy(X).float()
#
# pca = PCA(n_components=2)
# start = time.time()
# y = pca.fit_transform(X)
# end = time.time()
# print(end - start)