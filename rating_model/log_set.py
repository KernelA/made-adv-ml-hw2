"""Python logging config

"""

LOGGER_SETUP = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default_console_format': {
            'format': '%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s',
        }
    },
    'handlers': {
        'default_console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default_console_format',
            'stream': 'ext://sys.stdout'
        }
    },
    'loggers':
    {
        'rating_model.model': {
            'level': 'DEBUG',
            'handlers': ['default_console'],
            'propagate': False
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['default_console']
    }
}
