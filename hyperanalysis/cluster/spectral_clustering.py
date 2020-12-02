import torch

class SpectralClustering(object):

    def __init__(self, n_clusters: int = 8, n_components: int = None, n_init: int = 10, random_state: int = 0,
            gamma: float = 1.0, affinity: float = "rbf"):
        super(SpectralClustering, self).__init__()