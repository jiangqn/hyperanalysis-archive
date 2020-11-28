import torch
import numpy as np
import sys
sys.path.append("..")
from hyperanalysis.decomposition.truncated_svd import TruncatedSVD

path = "../embedding.npy"

X = np.load(path)
X = torch.from_numpy(X).float()

tsvd = TruncatedSVD(n_components=2)
X = tsvd.fit_transform(X)
print(X.shape)