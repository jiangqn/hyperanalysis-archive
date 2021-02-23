import torch
from torch import nn, optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from typing import Tuple
from hyperanalysis.linear_model.base import LinearClassifier

class SGDLogisticRegression(LinearClassifier):

    def __init__(self, lr: float = 0.001, batch_size: int = 64, epoches: int = 10,
                 l2_reg: float = 0, device = None) -> None:
        super(SGDLogisticRegression, self).__init__()
        self._lr = lr
        self._batch_size = batch_size
        self._epoches = epoches
        self._l2_reg = l2_reg
        self._device = device

    def _fit(self, X: torch.Tensor, y: torch.LongTensor) -> None:

        num, dim = X.size()
        num_categories = y.max().item() + 1

        device = X.device if self._device == None else X.device

        dataset = _ClassificationDataset(X, y)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            shuffle=True,
            pin_memory=True
        )

        self._model = _LogisticRegression(dim, num_categories)
        self._model = self._model.to(device)

        optimizer = optim.Adam(self._model.parameters(), lr=self._lr, weight_decay=self._l2_reg)
        criterion = nn.CrossEntropyLoss()

        min_loss = 1e9

        for epoch in range(self._epoches):
            for i, data in enumerate(dataloader):

                self._model.train()
                optimizer.zero_grad()

                feature, label = data
                feature, label = feature.to(device), label.to(device)

                logit = self._model(feature)
                loss = criterion(logit, label)

    def _predict(self, X: torch.Tensor) -> torch.LongTensor:
        pass

    def _fit_without_validate(self, X_train: torch.Tensor, y_train: torch.LongTensor) -> None:
        pass

    def _fit_with_validate(self, X_train: torch.Tensor, y_train: torch.LongTensor,
                           X_dev: torch.Tensor, y_dev: torch.LongTensor) -> None:
        pass

class _LogisticRegression(nn.Module):

    def __init__(self, hidden_size: int, num_categories: int) -> None:
        super(_LogisticRegression, self).__init__()
        self.linear = nn.Linear(hidden_size, num_categories)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(X)

class _ClassificationDataset(Dataset):

    def __init__(self, X: torch.Tensor, y: torch.LongTensor) -> None:
        self.X = X
        self.y = y
        self.num = y.size(0)

    def __len__(self) -> int:
        return self.num

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        return self.X[item], self.y[item]

def _eval(model: nn.Module, dataloader: DataLoader):
    pass