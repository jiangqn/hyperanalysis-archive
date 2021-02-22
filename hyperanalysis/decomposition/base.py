import torch
from typing import Tuple

class Decomposition(object):

    """
    The base class of dimension reduction models.
    """

    def __init__(self, n_components: int) -> None:
        super(Decomposition, self).__init__()
        self._n_components = n_components
        self._is_trained = False

    @property
    def n_components(self) -> int:
        return self._n_components

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    # def fit(self):
    #     raise NotImplementedError("The fit method of the Decomposition class is not implemented.")
    #
    # def transform(self):
    #     raise NotImplementedError("The transform method of the Decomposition class is not implemented.")
    #
    # def fit_transform(self):
    #     raise NotImplementedError("The fit_transform method of the Decomposition class is not implemented.")

class UnsupervisedDecomposition(Decomposition):

    """
    The base class of unsupervised dimension reduction models.
    """

    def __init__(self, n_components: int) -> None:
        super(UnsupervisedDecomposition, self).__init__(n_components)

    def fit(self, X: torch.Tensor) -> None:
        self._validate_inputs(X)
        self._fit(X)
        self._is_trained = True

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(X)
        assert self.is_trained
        return self._transform(X)

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: FloatTensor (num, dim)
        """
        self._validate_inputs(X)
        self._fit(X)
        self._is_trained = True
        return self._transform(X)

    def _fit(self, X: torch.Tensor) -> None:
        """
        :param X: FloatTensor (num, dim)
        """
        raise NotImplementedError("The _fit method is not implemented in the UnsupervisedDecomposition class.")

    def _transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: FloatTensor (num, dim)
        """
        raise NotImplementedError("The _transform method is not implemented in the UnsupervisedDecomposition class.")

    def _validate_inputs(self, X: torch.Tensor) -> None:
        """
        :param X: FloatTensor (num, dim)
        """
        assert isinstance(X, torch.Tensor), "The type of input X is wrong."
        assert len(X.size()) == 2, "This size of input X is wrong."

class SupervisedDecomposition(Decomposition):

    """
    The base class of supervised dimension reduction models.
    """

    def __init__(self, n_components: int) -> None:
        super(SupervisedDecomposition, self).__init__(n_components)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self._validate_inputs(X, y)
        self._fit(X, y)
        self._is_trained = True

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(X)
        assert self.is_trained
        return self._transform(X)

    def fit_transform(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(X, y)
        self._fit(X, y)
        self._is_trained = True
        return self._transform(X)

    def _fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        :param X: FloatTensor (num, dim)
        :param y: FloatTensor (num,) or LongTensor (num,)
        """
        raise NotImplementedError("The _fit method is not implemented in the SupervisedDecomposition class.")

    def _transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: FloatTensor (num, dim)
        """
        raise NotImplementedError("The _transform method is not implemented in the SupervisedDecomposition class.")

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

class CrossDecomposition(Decomposition):

    """
    The base class of cross dimension reduction models.
    """

    def __init__(self, n_components: int) -> None:
        super(CrossDecomposition, self).__init__(n_components)

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        self._validate_inputs(X, Y)
        self._fit(X, Y)
        self._is_trained = True

    def transform(self, X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(X, Y)
        assert self.is_trained
        return self._transform(X, Y)

    def fit_transform(self, X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(X, Y)
        self._fit(X, Y)
        self._is_trained = True
        return self._transform(X, Y)

    def _fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        :param X: FloatTensor (num, dim1)
        :param Y: FloatTensor (num, dim2)
        """
        raise NotImplementedError("The _fit method is not implemented in the CrossDecomposition class.")

    def _transform(self, X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param X: FloatTensor (num, dim1)
        :param Y: FloatTensor (num, dim2)
        :return X': FloatTensor (num, n_components)
        :return Y': FloatTensor (num, n_components)
        """
        raise NotImplementedError("The _transform method is not implemented in the CrossDecomposition class.")

    def _validate_inputs(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        :param X: FloatTensor (num, dim1)
        :param Y: FloatTensor (num, dim2)
        """
        assert isinstance(X, torch.Tensor), "The type of input X is wrong."
        assert len(X.size()) == 2, "This size of input X is wrong."
        assert isinstance(Y, torch.Tensor), "The type of input Y is wrong."
        assert len(Y.size()) == 2, "This size of input Y is wrong."
        assert X.size(0) == Y.size(0), "The num of X is not equal to Y."