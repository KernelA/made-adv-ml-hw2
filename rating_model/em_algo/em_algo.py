from collections import defaultdict
import logging
from itertools import repeat
import math
import os

import pandas as pd
import torch
from scipy import sparse
from torch import optim
from tqdm import trange
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch.utils.tensorboard import SummaryWriter

from ..torch_lr import LogisticRegressionTorch, Trainer
from ..rank_teams import get_player_skills, estimate_rank


class EMRatingModel:
    def __init__(self, *, em_num_iter: int,
                 lr: float, log_reg_num_iter: int, device, log_dir: str, checkpoint_dir: str):
        assert lr > 0
        assert em_num_iter > 0
        assert log_reg_num_iter > 0
        assert os.path.isdir(log_dir)
        assert os.path.isdir(checkpoint_dir)

        self._logger = logging.getLogger("rating_model.em_algo")
        if device.type == "cpu":
            self._logger.warning("Device for PyTorch training is CPU. Training may be slow than on GPU")
        self._logger.info("Will train logistic regression on %s", device)
        self._em_num_iter = em_num_iter
        self._log_reg_num_iter = log_reg_num_iter
        self._lr = lr
        self._device = device
        self._last_valid_metrics = None
        self._baseline_est = None
        self._log_dir = log_dir
        self._checkpoint_dir = checkpoint_dir

    def _init_data(self, sparse_features, target, players_info: pd.DataFrame,
                   baseline_est: dict, init_weights=None, bias=None):
        assert sparse_features.shape[0] == target.shape[0], "Number of samples is not equal number of targets"
        assert sparse.isspmatrix_coo(sparse_features), "Features must be in COO format"

        self._features = torch.sparse_coo_tensor(
            (sparse_features.row, sparse_features.col), sparse_features.data,
            size=sparse_features.shape, dtype=torch.get_default_dtype())
        assert self._features.shape == sparse_features.shape, "Sparse tensor features shapes is not equal an original features"
        self._target = torch.from_numpy(target).to(torch.get_default_dtype())
        # _pad_index is a fake index and it needed only for torch.take
        # Drop all loops
        # Values with this index is equal 0 always
        self._pad_index = target.shape[0]
        self._zeroing_mask = torch.ones_like(self._target, dtype=torch.bool)
        self._player_indices_in_team_by_round = self._build_player_team_round_indices(
            players_info)
        self._hidden_variables = self._target.clone()
        self._baseline_est = baseline_est
        self._best_metrics_value = None
        # Add fake 0 value for vectorize idexing operations
        self._predicted_proba = torch.zeros(self._hidden_variables.shape[0] + 1)
        self.model = LogisticRegressionTorch(self._features.shape[1])
        self.model.to(self._device)
        if init_weights is not None:
            self.model.init_pretrained({"weight": torch.tensor(init_weights), "bias": torch.tensor(bias)})
        else:
            self.model.init_xavier()

    def _clear_data(self):
        self._features = None
        self._target = None
        self._pad_index = None
        self._zeroing_mask = None
        self._hidden_variables = None
        self._player_indices_in_team_by_round = None
        self._last_valid_metrics = None
        self._baseline_est = None
        self._predicted_proba = None
        self._best_metrics = None
        self._best_metrics_value = None

    def _build_player_team_round_indices(self, player_info: pd.DataFrame) -> torch.LongTensor:
        self._logger.info("Building mask for zeroing hidden variables")
        player_indices_in_team_by_round = []

        # Number of digits
        base = 10 ** math.ceil(math.log10(player_info["team_id"].max()))
        self._logger.info("Use %d as base value for grouping", base)
        tour_team_id = pd.Series(pd.factorize(player_info["tour_id"] * base + player_info["team_id"])[0])

        max_length = -1

        for _, data in tour_team_id.groupby(tour_team_id):
            indices = data.index.to_list()
            player_indices_in_team_by_round.append(indices)
            max_length = max(max_length, len(indices))

        assert max_length > 0

        for i in trange(len(player_indices_in_team_by_round)):
            indices = player_indices_in_team_by_round[i]
            if (self._target[indices] > 0).any():
                self._zeroing_mask[indices] = False

            if len(indices) < max_length:
                player_indices_in_team_by_round[i].extend(
                    repeat(self._pad_index, max_length - len(indices)))

        return torch.LongTensor(player_indices_in_team_by_round)

    def best_checkpoint_file(self) -> str:
        return os.path.join(self._checkpoint_dir, "best_checkpoint.pt")

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

        self._hidden_variables[~torch.isfinite(self._hidden_variables)] = 0.0
        self._hidden_variables.masked_fill_(self._zeroing_mask, 0.0)

    @ torch.no_grad()
    def _expectation(self) -> None:
        """Estimate new expectation for hidden variables with new parameters

        Train logistic regression for predicting expectation of hidden variables
        """
        self._hidden_variables.fill_(0)
        features = self._features.to(self._device)
        pred_proba = self.model.predict_proba(features).cpu().view(-1)
        assert pred_proba.shape[0] == self._hidden_variables.shape[0]
        # This tensor contains fake value is equal to 0 at last position
        self._predicted_proba[:self._hidden_variables.shape[0]] = self.model.predict_proba(features).cpu().view(-1)
        self._update_hidden_values(self._predicted_proba)

    def fit(self, sparse_features, target, players_info: pd.DataFrame,
            skill_encoder: OneHotEncoder, test_team_ratings: pd.DataFrame, baseline_est: dict,
            init_weights: np.ndarray = None, init_bias: np.ndarray = None) -> dict:
        self._init_data(sparse_features, target, players_info, baseline_est, init_weights, init_bias)

        target_metrics = defaultdict(list)

        with SummaryWriter(self._log_dir, flush_secs=10) as writer:
            iterations = trange(self._em_num_iter, desc="EM algorithm")
            for step in iterations:
                iterations.set_description_str("E step")
                self._expectation()
                iterations.set_description_str("M step")
                self._maximization(step, writer)
                weights, _ = self.model.get_params()
                self._validate(step, weights.numpy(), skill_encoder, test_team_ratings, writer)

                target_metrics["em_iter"].append(step)
                for metric_name in self._last_valid_metrics:
                    target_metrics[metric_name].append(self._last_valid_metrics[metric_name])

        self._clear_data()

        return target_metrics

    def _validate(self, em_step: int, weights: np.ndarray, skill_encoder: OneHotEncoder,
                  test_team_ratings: pd.DataFrame, writer: SummaryWriter):
        player_ratings = get_player_skills(skill_encoder, weights)
        new_valid_metrics = estimate_rank(test_team_ratings, player_ratings)

        is_new_model_best = False

        self._logger.info("Absolute difference relative to baseline:")
        for metric_name in new_valid_metrics:
            diff = new_valid_metrics[metric_name] - self._baseline_est[metric_name]
            self._logger.info("%s %+.6f", metric_name, diff)
            if writer is not None:
                writer.add_scalar(f"Train/Diff_relative_to_baseline/{metric_name}", diff, global_step=em_step)

        if self._best_metrics_value is None:
            self._best_metrics_value = new_valid_metrics
            is_new_model_best = True
        elif all(new_valid_metrics[metric_name] > self._best_metrics_value[metric_name]
                 for metric_name in new_valid_metrics):
            self._best_metrics_value = new_valid_metrics
            is_new_model_best = True

        if is_new_model_best:
            checkpoint_path = self.best_checkpoint_file()
            self._logger.info("Save model state to '%s'", checkpoint_path)
            self.model.save_state(checkpoint_path)

        if self._last_valid_metrics is not None:
            self._logger.info("Absolute difference relative to previous params:")
            for metric_name in new_valid_metrics:
                diff = new_valid_metrics[metric_name] - self._last_valid_metrics[metric_name]
                self._logger.info("%s %+.6f", metric_name, diff)
                if writer is not None:
                    writer.add_scalar(f"Train/Diff relative to prev_params/{metric_name}", diff, global_step=em_step)

        self._last_valid_metrics = new_valid_metrics
        if writer is not None:
            writer.add_scalars("Train/Corr coefficients", self._last_valid_metrics, global_step=em_step)
        self._logger.info("Corr coefficients: %s", self._last_valid_metrics)

    def _maximization(self, em_step: int, writer: SummaryWriter):
        optimizer = optim.Adam(self.model.parameters(), self._lr)
        lr_sheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, min_lr=1e-6, patience=max(self._log_reg_num_iter // 4, 3))
        trainer = Trainer(model=self.model, em_step=em_step, optimizer=optimizer, num_iter=self._log_reg_num_iter,
                          device=self._device, sheduler=lr_sheduler, writer=writer)
        trainer.fit(self._features, self._hidden_variables.view(-1, 1))
        self.model.eval()
