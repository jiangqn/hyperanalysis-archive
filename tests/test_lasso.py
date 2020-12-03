import sys
sys.path.append("..")
import torch
import numpy as np
from hyperanalysis.linear_model.lasso import Lasso
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

X = X.cuda()
y = y.cuda()

lasso = Lasso(alpha=1.0)
lasso.fit(X, y)
print(lasso.coef_)
