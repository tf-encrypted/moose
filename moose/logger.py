import logging

_LOGGER = None


def get_logger():
    return _LOGGER


def set_logger(logger):
    global _LOGGER
    _LOGGER = logger


set_logger(logging.getLogger("runtime"))
logging.basicConfig()
