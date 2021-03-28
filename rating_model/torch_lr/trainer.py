import logging

from torch.nn import functional
from torch.utils import data
import tqdm


class SimpleDataloader(data.Dataset):
    def __init__(self, features, target):
        super().__init__()
        assert features.shape[0] == target.shape[0]
        self._features = features
        self._target = target

    def __len__(self):
        return self._target.shape[0]

    def __getitem__(self, item):
        return self._features[item], self._target[item]


class Trainer:
    def __init__(self, model, optimizer, num_iter: int, device):
        self._logger = logging.getLogger("rating_model.torch_trainer")
        self.model = model
        self._device = device
        self._optimizer = optimizer
        self._num_iter = num_iter
        self.model.to(device)

    def fit(self, features, target):
        device_features = features.to(self._device)
        device_target = target.to(self._device)
        self.model.train()

        for _ in tqdm.trange(self._num_iter):
            self._optimizer.zero_grad()

            predicted_logits = self.model(device_features)
            loss = functional.binary_cross_entropy_with_logits(
                predicted_logits, device_target, reduction="sum")

            loss.backward()
            self._optimizer.step()

            self._logger.info("Loss: %f", loss.item())
