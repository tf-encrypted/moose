import abc

import numpy as np
import struct

import torch.nn.functional as F
import torch.nn as nn

from pymoose import edsl
from pymoose.predictors import aes_predictor
from pymoose.predictors import predictor_utils


class NeuralNetwork(aes_predictor.AesPredictor, metaclass=abc.ABCMeta):
    def __init__(self, weights, biases, operations):
        super().__init__()
        self.weights = weights
        self.biases = biases
        self.operations = operations
        self.n_classes = np.shape(biases[-1])[0] # infer number of classes

    def apply_layer(self, input, i, fixedpoint_dtype):
        w = self.fixedpoint_constant(
            self.weights[i], plc=self.mirrored, dtype=fixedpoint_dtype
        )
        b = self.fixedpoint_constant(
            self.biases[i], plc=self.mirrored, dtype=fixedpoint_dtype
        )
        y = edsl.dot(input, w)
        z = edsl.add(y, b)
        return z

    def activation_fn(self, z, op):
        if op == "Sigmoid":
            activation_output = edsl.sigmoid(z)
            # There is a bug in edsl.shape
            # elif self.activation == "Relu":
            #     y_1_shape = edsl.slice(edsl.shape(x), begin=0, end=1)
            #     ones = edsl.ones(y_1_shape, dtype=edsl.float64)
            #     ones = edsl.cast(ones, dtype=fixedpoint_dtype)
            #     zeros = edsl.sub(ones, ones)
            #     activation_output = edsl.maximum([zeros, y_1])
        elif op == "Softmax":
            activation_output = edsl.softmax(z, axis=1, upmost_index=self.n_classes)
        else:
            raise ValueError("Invalid or unsupported activation function")
        return activation_output

    def neural_predictor_fn(self, x, fixedpoint_dtype):
        dense_layer_position = 0
        for op in self.operations:
            if op == "Gemm":
                x = self.apply_layer(x, dense_layer_position, fixedpoint_dtype)
                dense_layer_position += 1
            else:
                x = self.activation_fn(x, op)
        return x

    def predictor_factory(self, fixedpoint_dtype=predictor_utils.DEFAULT_FIXED_DTYPE):
        @edsl.computation
        def predictor(
            aes_data: edsl.Argument(
                self.alice, vtype=edsl.AesTensorType(dtype=fixedpoint_dtype)
            ),
            aes_key: edsl.Argument(self.replicated, vtype=edsl.AesKeyType()),
        ):
            x = self.handle_aes_input(aes_key, aes_data, decryptor=self.replicated)
            with self.replicated:
                y = self.neural_predictor_fn(x, fixedpoint_dtype)
            return self.handle_output(y, prediction_handler=self.bob)

        return predictor

    @classmethod
    def from_onnx(cls, model_proto):
        operations = predictor_utils.find_op_types_in_model_proto(model_proto)

        weights_data = predictor_utils.find_parameters_in_model_proto(
            model_proto, "weight", enforce=False
        )
        biases_data = predictor_utils.find_parameters_in_model_proto(
            model_proto, "bias", enforce=False
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
            weight = struct.unpack('f' * (dimentions[0] * dimentions[1]), weight)
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
            bias = struct.unpack('f' * dimentions[0], bias)
            bias = np.asarray(bias)
            biases.append(bias)

        return cls(weights, biases, operations)
