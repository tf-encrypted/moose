import argparse
import logging

from moose.compiler.edsl import HostPlacement
from moose.compiler.edsl import computation
from moose.compiler.edsl import function
from moose.compiler.edsl import save
from moose.logger import get_logger
from moose.runtime import RemoteRuntime
from moose.runtime import TestRuntime

parser = argparse.ArgumentParser(description="Run example")
parser.add_argument("--runtime", type=str, default="test")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

if args.verbose:
    get_logger().setLevel(level=logging.DEBUG)


inputter0 = HostPlacement(name="inputter0")
inputter1 = HostPlacement(name="inputter1")
aggregator = HostPlacement(name="aggregator")
outputter = HostPlacement(name="outputter")


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
        res = save(y, "y")

    return res


concrete_comp = my_comp.trace_func()


if __name__ == "__main__":

    if args.runtime == "test":
        runtime = TestRuntime(num_workers=len(concrete_comp.devices()))
    elif args.runtime == "remote":
        runtime = RemoteRuntime("./cluster-spec-main.yaml")
        assert len(runtime.executors) == len(concrete_comp.devices())
    else:
        raise ValueError(f"Unknown runtime '{args.runtime}'")

    runtime.evaluate_computation(
        computation=concrete_comp,
        placement_assignment={
            inputter0: runtime.executors[0],
            inputter1: runtime.executors[1],
            aggregator: runtime.executors[2],
            outputter: runtime.executors[3],
        },
    )

    print("Done")
