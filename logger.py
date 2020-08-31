import logging

_LOGGER = None


def set_logger(logger):
    global _LOGGER
    _LOGGER = logger


def get_logger():
    return _LOGGER


set_logger(logging.getLogger("runtime"))
logging.basicConfig()
