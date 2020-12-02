import sys
sys.path.append("..")
from hyperanalysis.decomposition.nmf import NMF

import torch
import numpy as np
import os

import time

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

path = "../embedding.npy"
X = np.load(path)
X = torch.from_numpy(X).cuda()
X = torch.abs(X)

nmf = NMF(n_components=10)
w, h = nmf.factorize(X)
print(w.shape, h.shape)
print(nmf.mse)