import sys
sys.path.append("..")
import torch
import numpy as np
from hyperanalysis.linear_model.ridge import Ridge
import hyperanalysis.utils.functional as F
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

path = "../embedding.npy"

X = np.load(path)
X = torch.from_numpy(X).float()

W = torch.randn(X.size(1), 1)
noise = torch.randn(X.size(0), 1)

y = X.matmul(W) + noise
y = y.squeeze(-1)

ridge = Ridge(alpha=2)
ridge.fit(X, y)

import math
print(math.sqrt(sum([x * x for x in ridge.coef_])))