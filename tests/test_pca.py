import sys
sys.path.append("..")
from hyperanalysis.decomposition.pca import PCA

import torch
import numpy as np
import os

from sklearn.decomposition.pca import PCA as sPCA

import time

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

path = "../embedding.npy"
X = np.load(path)
X = X - X.mean(axis=0)

spca = sPCA(n_components=2)

start = time.time()
sy = spca.fit_transform(X)
end = time.time()
print(end - start)
print(spca.explained_variance_)
print(spca.explained_variance_ratio_)

X = torch.from_numpy(X).float()
# X = X.cuda()

pca = PCA(n_components=2)
start = time.time()
y = pca.fit_transform(X)
end = time.time()
print(end - start)

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

y = y.cpu().numpy()

print(sy)
print(y)