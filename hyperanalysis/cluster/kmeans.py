import torch
import random
from hyperanalysis.utils.set_seed import set_seed
from hyperanalysis.utils.linalg import squared_euclidean_distance

class KMeans(object):

    def __init__(self, n_clusters: int = 8, init: str = "k-means++", n_init: int = 10,
                 max_iter: int = 300, tol: float = 1e-5, verbose: int = 0, random_state: int = 0) -> None:
        super(KMeans, self).__init__()
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        assert verbose in [0, 1]
        self.verbose = verbose
        set_seed(random_state)

    def fit(self, X: torch.FloatTensor) -> None:
        self._fit(X)

    def predict(self, X: torch.FloatTensor) -> torch.LongTensor:
        return self._predict(X)

    def transform(self, X: torch.FloatTensor) -> torch.FloatTensor:
        return self._transform(X)

    def fit_predict(self, X: torch.FloatTensor) -> torch.LongTensor:
        self._fit(X)
        return self._predict(X)

    def fit_transform(self, X: torch.FloatTensor) -> torch.FloatTensor:
        self._fit(X)
        return self._transform(X)

    def score(self, X) -> float:

        assert hasattr(self, "cluster_centers_")
        assert hasattr(self, "mean_min_squared_distance_")

        assert len(X.size()) == 2

        min_squared_distance = squared_euclidean_distance(X, self.cluster_centers_).min(dim=1)[0]
        mean_min_squared_distance = min_squared_distance.mean().item()
        return mean_min_squared_distance

    def _init(self, X: torch.FloatTensor) -> None:

        assert self.init in ["k-means++", "random"]
        assert len(X.size()) == 2

        num = X.size(0)
        assert num >= self.n_clusters

        device = X.device

        if self.init == "k-means++":
            index = random.randint(0, num - 1)
            cluster_centers_ = X[index: index + 1]
            for i in range(1, self.n_clusters):
                minimal_distance = squared_euclidean_distance(X, cluster_centers_).min(dim=1)[0]
                index = torch.multinomial(minimal_distance, 1, replacement=False).item()
                cluster_centers_ = torch.cat([cluster_centers_, X[index: index + 1]], dim=0)
            self.cluster_centers_ = cluster_centers_

        else:   # self.init == "random"
            cluster_center_index = torch.randperm(num, device=device)[0:self.n_clusters]
            self.cluster_centers_ = X[cluster_center_index]

    def _fit(self, X: torch.FloatTensor) -> None:

        cluster_centers_ = None
        mean_min_squared_distance_ = None

        for i in range(1, self.n_init + 1):

            if self.verbose == 1:
                print("Init #%d" % i)

            self._init_fit(X)

            if mean_min_squared_distance_ == None or self.mean_min_squared_distance_ < mean_min_squared_distance_:
                cluster_centers_ = self.cluster_centers_
                mean_min_squared_distance_ = self.mean_min_squared_distance_

        self.cluster_centers_ = cluster_centers_
        self.mean_min_squared_distance_ = mean_min_squared_distance_

        if self.verbose == 1:
            print("Finish.")
            print("Best mean_min_squared_distance: %.4f" % self.mean_min_squared_distance_)


    def _init_fit(self, X: torch.FloatTensor) -> None:

        self._init(X)

        last_mean_min_squared_distance = None
        for i in range(1, self.max_iter + 1):

            min_squared_distance, min_squared_distance_index = squared_euclidean_distance(X, self.cluster_centers_).min(dim=1)
            mean_min_squared_distance = min_squared_distance.mean().item()

            if last_mean_min_squared_distance == None or last_mean_min_squared_distance - mean_min_squared_distance >= self.tol:
                for j in range(self.n_clusters):
                    self.cluster_centers_[j] = X[min_squared_distance_index == j].mean(dim=0)
                if self.verbose == 1:
                    print("\t[step %d] [mean_min_squared_distance %.4f]" % (i, mean_min_squared_distance))
                last_mean_min_squared_distance = mean_min_squared_distance
            else:
                if self.verbose == 1:
                    print("\tInit Finish.")
                break

        self.mean_min_squared_distance_ = last_mean_min_squared_distance

    def _predict(self, X: torch.FloatTensor) -> torch.LongTensor:

        assert hasattr(self, "cluster_centers_")
        assert hasattr(self, "mean_min_squared_distance_")

        assert len(X.size()) == 2

        label = squared_euclidean_distance(X, self.cluster_centers_).min(dim=1)[1]
        return label

    def _transform(self, X: torch.FloatTensor) -> torch.FloatTensor:

        assert hasattr(self, "cluster_centers_")
        assert hasattr(self, "mean_min_squared_distance_")

        assert len(X.size()) == 2

        distance = torch.sqrt(squared_euclidean_distance(X, self.cluster_centers_)).min(dim=1)[0]
        return distance