import torch

class Cluster(object):
    """
    The base class of clustering models.
    """

    def __init__(self, n_clusters: int) -> None:
        super(Cluster, self).__init__()
        self._n_clusters = n_clusters
        self._is_trained = False

    @property
    def n_clusters(self) -> int:
        return self._n_clusters

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def fit(self, X: torch.Tensor) -> None:
        """
        :param X: FloatTensor (num, dim)
        """
        self._validate_inputs(X)
        self._fit(X)
        self._is_trained = True

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: FloatTensor (num, dim)
        :return : LongTensor (num,)
        """
        self._validate_inputs(X)
        assert self.is_trained
        return self._predict(X)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: FloatTensor (num, dim)
        :return : FloatTensor (num, n_clusters)
        """
        self._validate_inputs(X)
        assert self.is_trained
        return self._transform(X)

    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: FloatTensor (num, dim)
        """
        self._validate_inputs(X)
        self._fit(X)
        self._is_trained = True
        return self._predict(X)

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: FloatTensor (num, dim)
        """
        self._validate_inputs(X)
        self._fit(X)
        self._is_trained = True
        return self._transform(X)

    def _fit(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: FloatTensor (num, dim)
        """
        raise NotImplementedError("The _fit method is not implemented in the Cluster class.")

    def _predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: FloatTensor (num, dim)
        """
        raise NotImplementedError("The _predict method is not implemented in the Cluster class.")

    def _transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: FloatTensor (num, dim)
        """
        raise NotImplementedError("The _transform method is not implemented in the Cluster class.")

    def _validate_inputs(self, X: torch.Tensor) -> None:
        assert isinstance(X, torch.Tensor), "The type of input X is wrong."
        assert len(X.size()) == 2, "This size of input X is wrong."