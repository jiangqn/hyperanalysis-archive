import sys
sys.path.append("..")
import torch
import numpy as np
from hyperanalysis.cluster.kmeans import KMeans
import hyperanalysis.utils.functional as F
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

path = "../embedding.npy"

X = np.load(path)
X = torch.from_numpy(X).cuda()

kmeans = KMeans(n_clusters=8, verbose=0)
kmeans.fit(X)
print(kmeans.predict(X).shape)
print(kmeans.transform(X).shape)