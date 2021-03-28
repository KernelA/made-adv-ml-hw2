import logging
from itertools import repeat

import torch
from torch import optim
import pandas as pd
from scipy import sparse
from tqdm import tqdm, trange

from ..torch_lr import Trainer, LogisticRegressionTorch


class EMRatingModel:
    def __init__(self, sparse_features, target, players_info: pd.DataFrame, em_num_iter: int,
                 lr: float, log_reg_num_iter: int, device):
        assert sparse_features.shape[0] == target.shape[0]
        assert sparse.isspmatrix_coo(sparse_features), "Features must be in COO format"

        self._logger = logging.getLogger("rating_model.em_algo")
        self._features = torch.sparse_coo_tensor(
            (sparse_features.row, sparse_features.col), sparse_features.data,
            size=sparse_features.shape, dtype=torch.get_default_dtype()).to(device)
        assert self._features.shape == sparse_features.shape, "Features shapes is not equal"
        self._target = torch.from_numpy(target).to(torch.get_default_dtype())
        # _pad_index это фейковый индекс и нужен только для того чтобы использовать функцию np.take
        # значение по этому индексу всегда равно 0
        self._pad_index = target.shape[0]
        self._zeroing_mask = torch.zeros_like(self._target, dtype=torch.int8)
        self._player_indices_in_team_by_round = self._build_player_team_round_indices(
            players_info).to(device)
        self._hidden_variables = torch.zeros_like(self._target, device=device)
        self._em_num_iter = em_num_iter
        self._log_reg_num_iter = log_reg_num_iter
        self._lr = lr
        self.model = LogisticRegressionTorch(self._features.shape[1])
        self.model.to(device)
        self.model.init_xavier()
        self._device = device

    def _build_player_team_round_indices(self, player_info):
        self._logger.info("Build indices masks")
        player_indices_in_team_by_round = []

        tour_team_id = pd.Series((player_info["tour_id"].astype(
            str) + " " + player_info["team_id"].astype(str)).factorize()[0])

        max_length = -1

        for _, data in tour_team_id.groupby(tour_team_id):
            indices = data.index.to_list()
            player_indices_in_team_by_round.append(indices)
            max_length = max(max_length, len(indices))

        assert max_length > 0

        for i in trange(len(player_indices_in_team_by_round)):
            indices = player_indices_in_team_by_round[i]
            if len(indices) < max_length:
                if (self._target[indices] > 0).any():
                    self._zeroing_mask[indices] = 1

                player_indices_in_team_by_round[i].extend(
                    repeat(self._pad_index, max_length - len(indices)))

        return torch.LongTensor(player_indices_in_team_by_round)

    @torch.no_grad()
    def _update_hidden_values(self, predicted_proba):
        predicted_proba_by_groups = torch.take(
            predicted_proba, self._player_indices_in_team_by_round)
        predicted_proba_by_groups /= (1 - torch.prod(1 -
                                                     predicted_proba_by_groups, dim=1).reshape(-1, 1))

        for i, index in enumerate(self._player_indices_in_team_by_round):
            not_fake_mask = index != self._pad_index
            not_fake_indices = index[not_fake_mask]
            self._hidden_variables[not_fake_indices] = predicted_proba_by_groups[i, not_fake_mask]

        self._hidden_variables = torch.nan_to_num(self._hidden_variables)

    @torch.no_grad()
    def _expectation(self, regression_params, bias):
        """Estimate new expectation for hidden variables with new parameters
        """
        self._logger.info("Expectation step")
        self._hidden_variables.fill_(0)
        predicted_proba = self._features @ regression_params.view(-1)
        predicted_proba += bias
        # Add fake value for vectorizing idexing operations
        predicted_proba = torch.cat((predicted_proba, torch.tensor([0]).to(self._device)))
        self._update_hidden_values(predicted_proba)

    def fit(self):
        for _ in trange(self._em_num_iter, desc="EM algorithm"):
            log_regr_params, bias = self.model.get_params()
            self._expectation(log_regr_params, bias)
            self._maximization()

    def _maximization(self):
        optimizer = optim.Adam(self.model.parameters(), self._lr)
        trainer = Trainer(self.model, optimizer, self._log_reg_num_iter, self._device)
        trainer.fit(self._features, self._hidden_variables.reshape(-1, 1))
        self.model.eval()
