import logging.config

from .data_model import TeamResults
from .constants import *
from .log_set import LOGGER_SETUP

logging.config.dictConfig(LOGGER_SETUP)
