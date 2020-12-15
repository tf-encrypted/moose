import pickle

from moose.logger import get_logger


def serialize_computation(computation):
    return pickle.dumps(computation)


def deserialize_computation(bytes_stream):
    computation = pickle.loads(bytes_stream)
    get_logger().debug(computation)
    return computation
