from typing import Tuple

import torch
from torch import nn
from ..constants import PICKLE_PROTOCOL


class LogisticRegressionTorch(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.lin_layer = nn.Linear(in_features, 1)

    def forward(self, features) -> torch.Tensor:
        """Return logit not probability
        """
        return self.lin_layer(features)

    def init_xavier(self) -> None:
        torch.nn.init.xavier_normal_(self.lin_layer.weight)

    def init_pretrained(self, state_dict: dict) -> None:
        self.lin_layer.load_state_dict(state_dict)

    def predict_proba(self, features) -> torch.Tensor:
        return torch.sigmoid(self.forward(features))

    @torch.no_grad()
    def get_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        linear_weights = self.lin_layer.weight[0].detach().clone().cpu()
        bias = self.lin_layer.bias.detach().clone().cpu()
        return linear_weights, bias

    def save_state(self, path_to_checkpoint) -> None:
        with open(path_to_checkpoint, "wb") as checkpoint_file:
            torch.save(self.state_dict(), checkpoint_file, pickle_protocol=PICKLE_PROTOCOL)

    @staticmethod
    def load_from_checkpoint(model, path_to_checkpoint) -> None:
        with open(path_to_checkpoint, "wb") as checkpoint_file:
            state = torch.load(checkpoint_file)
            model.load_state_dict(state)
