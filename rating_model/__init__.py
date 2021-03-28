import logging.config

from .data_model import TeamResults
from .constants import *
from .log_set import LOGGER_SETUP
from .torch_lr import LogisticRegressionTorch, Trainer
from .em_algo import EMRatingModel

logging.config.dictConfig(LOGGER_SETUP)
