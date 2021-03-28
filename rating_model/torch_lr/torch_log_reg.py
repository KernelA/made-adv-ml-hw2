from typing import Tuple

import torch
from torch import nn


class LogisticRegressionTorch(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.lin_module = nn.Linear(in_features, 1)

    def forward(self, features):
        """Return logit not probability
        """
        return self.lin_module(features)

    def init_xavier(self):
        torch.nn.init.xavier_normal_(self.lin_module.weight)

    def predict_proba(self, features):
        return torch.sigmoid(self.forward(features))

    @torch.no_grad()
    def get_params(self) -> Tuple[torch.Tensor]:
        linear_weights = self.lin_module.weight[0].detach().clone()
        bias = self.lin_module.bias.detach().clone()
        return linear_weights, bias
