import sys
sys.path.append("..")
import torch
import numpy as np
from hyperanalysis.representation_similarity.cka import CKA
import hyperanalysis.utils.functional as F
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

path = "../embedding.npy"

X = np.load(path)
X = torch.from_numpy(X).float().cuda()

Y = X[:, 100:200]
X = X[:, 0:100]

import time
cka = CKA(kernel="rbf")
start = time.time()
print(cka.score(X, Y))
end = time.time()
print(end - start)