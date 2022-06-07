import struct
from enum import Enum

import numpy as np

import pymoose as pm
from pymoose.predictors import predictor
from pymoose.predictors import predictor_utils


class Activation(Enum):
    IDENTITY = 1
    SIGMOID = 2
    SOFTMAX = 3
    RELU = 4


class NeuralNetwork(predictor.Predictor):
    def __init__(self, weights, biases, activations):
        super().__init__()
        self.weights = weights
        self.biases = biases
        self.activations = activations
        self.n_classes = np.shape(biases[-1])[0]  # infer number of classes

    def apply_layer(self, input, i, fixedpoint_dtype):
        w = self.fixedpoint_constant(
            self.weights[i], plc=self.mirrored, dtype=fixedpoint_dtype
        )
        b = self.fixedpoint_constant(
            self.biases[i], plc=self.mirrored, dtype=fixedpoint_dtype
        )
        y = pm.dot(input, w)
        z = pm.add(y, b)
        return z

    def activation_fn(self, z, i):
        activation = self.activations[i]
        if activation == Activation.SIGMOID:
            activation_output = pm.sigmoid(z)
        elif activation == Activation.RELU:
            z_shape = pm.shape(z)
            with self.bob:
                zeros = pm.zeros(z_shape, dtype=predictor_utils.DEFAULT_FLOAT_DTYPE)
                zeros = pm.cast(zeros, dtype=predictor_utils.DEFAULT_FIXED_DTYPE)
            activation_output = pm.maximum([zeros, z])
        elif activation == Activation.SOFTMAX:
            activation_output = pm.softmax(z, axis=1, upmost_index=self.n_classes)
        elif activation == Activation.IDENTITY:
            activation_output = z
        else:
            raise ValueError("Invalid or unsupported activation function")

        return activation_output

    def neural_predictor_fn(self, x, fixedpoint_dtype):
        num_layers = len(self.weights)
        for i in range(num_layers):
            x = self.apply_layer(x, i, fixedpoint_dtype)
            x = self.activation_fn(x, i)

        return x

    def __call__(self, x, fixedpoint_dtype=predictor_utils.DEFAULT_FIXED_DTYPE):
        return self.neural_predictor_fn(x, fixedpoint_dtype)

    @classmethod
    def from_onnx(cls, model_proto):
        # extract activations from operations
        operations = predictor_utils.find_op_types_in_model_proto(model_proto)
        activations = []
        for i in range(len(operations)):
            if operations[i] == "Sigmoid":
                activations.append(Activation.SIGMOID)
            elif operations[i] == "Softmax":
                activations.append(Activation.SOFTMAX)
            elif operations[i] == "Relu":
                activations.append(Activation.RELU)
            # PyTorch
            if i > 0:
                if operations[i] == "Gemm" and operations[i - 1] == "Gemm":
                    activations.append(Activation.IDENTITY)
            # TF Keras
            if i > 2:
                if (
                    operations[i] == "Add"
                    and operations[i - 1] == "MatMul"
                    and operations[i - 2] == "Add"
                    and operations[i - 3] == "MatMul"
                ):
                    activations.append(Activation.IDENTITY)

        # PyTorch: weight, bias; TF Keras: MatMul, BiasAdd
        weights_data = predictor_utils.find_parameters_in_model_proto(
            model_proto, ["weight", "MatMul"], enforce=False
        )
        biases_data = predictor_utils.find_parameters_in_model_proto(
            model_proto, ["bias", "BiasAdd"], enforce=False
        )
        weights = []
        for weight in weights_data:
            dimentions = weight.dims
            assert weight is not None
            if weight.data_type != 1:  # FLOATS
                raise ValueError(
                    "Neural Network Weights must be of type FLOATS, found other."
                )
            weight = weight.raw_data
            # decode bytes object
            weight = struct.unpack("f" * (dimentions[0] * dimentions[1]), weight)
            weight = np.asarray(weight)
            weight = weight.reshape(dimentions[0], dimentions[1]).T
            weights.append(weight)

        biases = []
        for bias in biases_data:
            dimentions = bias.dims
            assert bias is not None
            if bias.data_type != 1:  # FLOATS
                raise ValueError(
                    "Neural network biases must be of type FLOATS, found other."
                )
            bias = bias.raw_data
            bias = struct.unpack("f" * dimentions[0], bias)
            bias = np.asarray(bias)
            biases.append(bias)

        # TF Keras onnx graph stores weights and biases in reversed order
        # I.e.: from last to first layer
        if "tf" in model_proto.producer_name:
            weights = weights[::-1]
            biases = biases[::-1]
            # TF Keras weights need to be transposed
            weights = [item.T for item in weights]

        # `n_features` arg
        model_input = model_proto.graph.input[0]
        input_shape = predictor_utils.find_input_shape(model_input)
        assert len(input_shape) == 2
        n_features = input_shape[1].dim_value

        first_layer_weights_shape = weights[0].shape

        if n_features != first_layer_weights_shape[0]:
            raise ValueError(
                f"In the ONNX file, the input shape has {n_features} "
                "features and the shape of the weights for the first "
                f"layer is: {first_layer_weights_shape}. Validate you set "
                "correctly the `initial_types` when converting "
                "your model to ONNX."
            )

        return cls(weights, biases, activations)
