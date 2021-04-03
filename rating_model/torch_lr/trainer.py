import logging

import torch
from torch.nn import functional
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
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
    def __init__(self, *, em_step: int, model, optimizer, num_iter: int, device, sheduler=None, writer: SummaryWriter = None):
        self._logger = logging.getLogger("rating_model.torch_trainer")
        self.model = model
        self._device = device
        self._optimizer = optimizer
        self._num_iter = num_iter
        self.model.to(device)
        self._sheduler = sheduler
        self._writer = writer
        self._em_step = em_step

    def fit(self, features, target):
        device_features = features.to(self._device)
        device_target = target.to(self._device)
        self.model.train()

        iterator = tqdm.trange(self._num_iter, desc="Train logistic regression")

        for epoch in iterator:
            self._optimizer.zero_grad()

            predicted_logits = self.model(device_features)

            loss = functional.binary_cross_entropy_with_logits(
                predicted_logits, device_target, reduction="mean")

            mae_loss = functional.l1_loss(torch.sigmoid(predicted_logits), device_target)

            loss.backward()
            loss_val = loss.item()
            metrics = {"Binary cross entropy": loss_val, "MAE": mae_loss.item()}
            self._optimizer.step()

            iterator.set_postfix(metrics)
            self._writer.add_scalars(f"Train/EM step {self._em_step}", metrics, global_step=epoch)

            if self._sheduler is not None:
                self._sheduler.step(loss_val)
