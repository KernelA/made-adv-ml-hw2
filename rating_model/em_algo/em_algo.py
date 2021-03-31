import logging
from itertools import repeat

import pandas as pd
import torch
from scipy import sparse
from torch import optim
from tqdm import trange
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from ..torch_lr import LogisticRegressionTorch, Trainer
from ..rank_teams import get_player_skills, estimate_rank


class EMRatingModel:
    def __init__(self, *, em_num_iter: int,
                 lr: float, log_reg_num_iter: int, device):
        assert lr > 0
        assert em_num_iter > 0
        assert log_reg_num_iter > 0

        self._logger = logging.getLogger("rating_model.em_algo")
        if device.type == "cpu":
            self._logger.warning("Device for PyTorch training is CPU. Training may be very slow")
        self._logger.info("Will train logistic regression on %s", device)
        self._em_num_iter = em_num_iter
        self._log_reg_num_iter = log_reg_num_iter
        self._lr = lr
        self._device = device

    def _init_data(self, sparse_features, target, players_info: pd.DataFrame):
        assert sparse_features.shape[0] == target.shape[0], "Number of samples is not equal number of targets"
        assert sparse.isspmatrix_coo(sparse_features), "Features must be in COO format"

        self._features = torch.sparse_coo_tensor(
            (sparse_features.row, sparse_features.col), sparse_features.data,
            size=sparse_features.shape, dtype=torch.get_default_dtype())
        assert self._features.shape == sparse_features.shape, "Sparse tensor features shapes is not equal an original features"
        self._target = torch.from_numpy(target).to(torch.get_default_dtype())
        # _pad_index это фейковый индекс и нужен только для того чтобы использовать функцию torch.take
        # для вектаризации всех вычислений и избавления от циклов
        # значение по этому индексу всегда равно 0
        self._pad_index = target.shape[0]
        self._zeroing_mask = torch.zeros_like(self._target, dtype=torch.bool)
        self._player_indices_in_team_by_round = self._build_player_team_round_indices(
            players_info)
        self._hidden_variables = self._target.clone()
        self.model = LogisticRegressionTorch(self._features.shape[1])
        self.model.to(self._device)
        self.model.init_xavier()

    def _clear_data(self):
        self._features = None
        self._target = None
        self._zeroing_mask = None
        self._hidden_variables = None
        self._player_indices_in_team_by_round = None

    def _build_player_team_round_indices(self, player_info) -> torch.LongTensor:
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
                    self._zeroing_mask[indices] = True

                player_indices_in_team_by_round[i].extend(
                    repeat(self._pad_index, max_length - len(indices)))

        return torch.LongTensor(player_indices_in_team_by_round)

    @torch.no_grad()
    def _update_hidden_values(self, predicted_proba) -> None:
        """Update values of hidden variables
        """
        predicted_proba_by_groups = torch.take(
            predicted_proba, self._player_indices_in_team_by_round)
        predicted_proba_by_groups /= (1 - torch.prod(1 -
                                                     predicted_proba_by_groups, dim=1, keepdim=True))

        for i, index in enumerate(self._player_indices_in_team_by_round):
            not_fake_mask = index != self._pad_index
            not_fake_indices = index[not_fake_mask]
            self._hidden_variables[not_fake_indices] = predicted_proba_by_groups[i, not_fake_mask]

        self._hidden_variables.nan_to_num_(0)
        self._hidden_variables.masked_fill_(self._zeroing_mask, 0.0)

    @ torch.no_grad()
    def _expectation(self) -> None:
        """Estimate new expectation for hidden variables with new parameters

        Train logistic regression for predicting expectation of hidden variables
        """
        self._hidden_variables.fill_(0)
        features = self._features.to(self._device)
        predicted_proba = self.model.predict_proba(features).cpu().view(-1)
        # Add fake value for vectorizing idexing operations
        predicted_proba = torch.cat(
            (predicted_proba, torch.tensor([0], dtype=predicted_proba.dtype)))
        self._update_hidden_values(predicted_proba)

    def fit(self, sparse_features, target, players_info: pd.DataFrame,
            skill_encoder: OneHotEncoder, test_team_ratings: pd.DataFrame):
        self._init_data(sparse_features, target, players_info)

        iterations = trange(self._em_num_iter, desc="EM algorithm")
        for _ in iterations:
            iterations.set_description_str("M step")
            self._maximization()
            iterations.set_description_str("E step")
            self._expectation()
            weights, _ = self.model.get_params()
            self._validate(weights.numpy(), skill_encoder, test_team_ratings)

        self._clear_data()

    def _validate(self, weights: np.ndarray, skill_encoder: OneHotEncoder, test_team_ratings: pd.DataFrame):
        player_ratings = get_player_skills(skill_encoder, weights)
        player_ratings.sort_values("skill", inplace=True)
        self._logger.info("Corr coef: %s", estimate_rank(test_team_ratings, player_ratings))

    def _maximization(self):
        optimizer = optim.Adam(self.model.parameters(), self._lr)
        trainer = Trainer(self.model, optimizer, self._log_reg_num_iter, self._device)
        trainer.fit(self._features, self._hidden_variables.reshape(-1, 1))
        self.model.eval()
