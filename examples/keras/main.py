import argparse
import logging

from moose.choreography.grpc import Choreographer as GrpcChoreographer
from moose.edsl import computation
from moose.edsl import function
from moose.edsl import host_placement
from moose.edsl import save
from moose.edsl import trace
from moose.logger import get_logger
from moose.testing import TestRuntime

parser = argparse.ArgumentParser(description="Run example")
parser.add_argument("--runtime", type=str, default="test")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

if args.verbose:
    get_logger().setLevel(level=logging.DEBUG)


inputter0 = host_placement(name="inputter0")
inputter1 = host_placement(name="inputter1")
aggregator = host_placement(name="aggregator")
outputter = host_placement(name="outputter")


@function(output_type="numpy.ndarray")
def load_data():
    import numpy

    return numpy.array([5])


@function(output_type="tf.keras.model")
def load_model():
    import tensorflow as tf

    model = tf.keras.models.Sequential([tf.keras.layers.Dense(1)])
    model.build(input_shape=[1, 1])
    return model


@function
def get_weights(model):
    return model.trainable_weights


@function(output_type="numpy.ndarray")
def model_predict(model, input, weights):
    return model.predict(input)


@computation
def my_comp():

    with inputter0:
        model = load_model()
        weights = get_weights(model)

    with inputter1:
        x = load_data()

    with aggregator:
        y = model_predict(model, x, weights)

    with outputter:
        res = save("y", y)

    return res


concrete_comp = trace(my_comp)


if __name__ == "__main__":

    if args.runtime == "test":
        runtime = TestRuntime()
    elif args.runtime == "remote":
        runtime = GrpcChoreographer()
    else:
        raise ValueError(f"Unknown runtime '{args.runtime}'")

    runtime.evaluate_computation(
        computation=concrete_comp,
        placement_instantiation={
            inputter0: "inputter0:50000",
            inputter1: "inputter1:50000",
            aggregator: "aggregator:50000",
            outputter: "outputter:50000",
        },
    )

    print("Done")
