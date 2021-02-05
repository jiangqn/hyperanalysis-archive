# hyperanalysis
An efficient tensor data analysis toolkit based on PyTorch.

## Installation

### From Pypi

```
pip install hyperanalysis
```

### From Source

```
python setup.py install
```

## Introduction

There are four components in this tookit:

- Dimension reduction and visualization of data.
- Clustering and quantization of data.
- Similarity and distance.
- Linear models.

### Dimension Reduction and Visualization of Data

This component aims to reduce the dimension of data especially to 2d,
to get insights of the distribution of data (with respect to certain features).

In this component, we implemented some common dimension reduction methods,
which could be classified by supervision and linearity as follows.

| |  Unsupervised   | Supervised  |
|  ----  |  ----  | ----  |
| Linear | PCA, SVD, NMF  | LDA, LRA |
| Non-Linear | kPCA, MDS, tSNE  | QDA |

### Clustering and Quantization of Data

### Similarity and Distance

### Linear Models