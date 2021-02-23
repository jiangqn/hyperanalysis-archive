import torch

class LinearModel(object):

    def __init__(self) -> None:
        super(LinearModel, self).__init__()
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _validate_inputs(self, X: torch.Tensor, y: torch.Tensor = None) -> None:
        """
        :param X: FloatTensor (num, dim)
        :param y: FloatTensor (num,) or LongTensor (num,)
        """
        assert isinstance(X, torch.Tensor), "The type of input X is wrong."
        assert len(X.size()) == 2, "This size of input X is wrong."
        if y is not None:
            assert isinstance(y, torch.Tensor), "The type of input y is wrong."
            assert len(y.size()) == 1, "This size of input y is wrong."
            assert X.size(0) == y.size(0), "The num of X is not equal to y."

class LinearClassifier(LinearModel):

    def __init__(self) -> None:
        super(LinearClassifier, self).__init__()

    def fit(self, X: torch.Tensor, y: torch.LongTensor) -> None:
        """
        :param X: FloatTensor (num, dim)
        :param y: LongTensor (num,)
        """
        self._validate_inputs(X, y)
        self._fit(X, y)
        self._is_trained = True

    def predict(self, X: torch.Tensor) -> torch.LongTensor:
        """
        :param X: FloatTensor (num, dim)
        :return : LongTensor (num,)
        """
        self._validate_inputs(X)
        assert self.is_trained
        return self._predict(X)

    def fit_predict(self, X: torch.Tensor, y: torch.LongTensor) -> torch.LongTensor:
        """
        :param X: FloatTensor (num, dim)
        :param y: LongTensor (num,)
        :return : LongTensor (num,)
        """
        self._validate_inputs(X, y)
        self._fit(X, y)
        self._is_trained = True
        return self._predict(X)

    def _fit(self, X: torch.Tensor, y: torch.LongTensor) -> None:
        raise NotImplementedError("The _fit method is not implemented in the LinearClassifier class.")

    def _predict(self, X: torch.Tensor) -> torch.LongTensor:
        raise NotImplementedError("The _predict method is not implemented in the LinearClassifier class.")

class LinearRegressor(LinearModel):
    
    def __init__(self) -> None:
        super(LinearRegressor, self).__init__()

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        :param X: FloatTensor (num, dim)
        :param y: Tensor (num,)
        """
        self._validate_inputs(X, y)
        self._fit(X, y)
        self._is_trained = True

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: FloatTensor (num, dim)
        :return : Tensor (num,)
        """
        self._validate_inputs(X)
        assert self.is_trained
        return self._predict(X)

    def fit_predict(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param X: FloatTensor (num, dim)
        :param y: Tensor (num,)
        :return : Tensor (num,)
        """
        self._validate_inputs(X, y)
        self._fit(X, y)
        self._is_trained = True
        return self._predict(X)

    def _fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        raise NotImplementedError("The _fit method is not implemented in the LinearRegressor class.")

    def _predict(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("The _predict method is not implemented in the LinearRegressor class.")