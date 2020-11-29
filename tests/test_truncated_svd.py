import torch
import numpy as np
import sys
sys.path.append("..")
from hyperanalysis.decomposition.truncated_svd import TruncatedSVD
from sklearn.decomposition.truncated_svd import TruncatedSVD as sTruncatedSVD

path = "../embedding.npy"

X = np.load(path)

stsvd = sTruncatedSVD(n_components=2)
sX = stsvd.fit_transform(X)

X = torch.from_numpy(X).float()

tsvd = TruncatedSVD(n_components=2)
X = tsvd.fit_transform(X).numpy()