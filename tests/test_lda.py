import torch
from hyperanalysis.discriminant_analysis.lda import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

X = torch.from_numpy(X)
y = torch.from_numpy(y)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit_transform(X, y).numpy()

plt.figure()
colors = ["navy", "turquoise", "darkorange"]
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("LDA of IRIS dataset")

plt.show()