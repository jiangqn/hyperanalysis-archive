import sys
sys.path.append("..")
import torch
import numpy as np
from hyperanalysis.representation_similarity.svcca import SVCCA

path = "../embedding.npy"

X = np.load(path)
X = torch.from_numpy(X).float()

Y = X[:, 100:250]
X = X[:, 0:100]

svcca = SVCCA()
print(svcca.score(X, Y))