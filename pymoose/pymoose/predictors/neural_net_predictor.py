import abc
import struct

import numpy as np

from pymoose import edsl
from pymoose.predictors import aes_predictor
from pymoose.predictors import predictor_utils


class NeuralNetwork(aes_predictor.AesPredictor, metaclass=abc.ABCMeta):
    def __init__(self, weights, biases, activations):
        super().__init__()
        self.weights = weights
        self.biases = biases
        self.activations = activations

    @classmethod
    @abc.abstractmethod
    def from_onnx(cls, model_proto):
        pass

    @abc.abstractmethod
    def apply_layer(self, input, i, fixedpoint_dtype):
        pass

    @classmethod
    @abc.abstractmethod
    def activation_fn(self, z, i=None):
        pass

    @abc.abstractmethod
    def predictor_factory(self, fixedpoint_dtype=predictor_utils.DEFAULT_FIXED_DTYPE):
        pass

    @abc.abstractmethod
    def neural_predictor_fn(self, x, fixedpoint_dtype):
        pass


class FullyConnectedNeuralNetwork(NeuralNetwork):
    @classmethod
    def from_onnx(cls, model_proto):
        # extract activations from operations
        operations = predictor_utils.find_op_types_in_model_proto(model_proto)
        activations = []
        for i in range(len(operations)):
            if operations[i] != "Gemm":
                activations.append(operations[i])
            if i > 0:
                if operations[i] == "Gemm" and operations[i - 1] == "Gemm":
                    activations.append("Identity")
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

        return cls(weights, biases, activations)

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

    def activation_fn(self, z, i):
        activation = self.activations[i]
        if activation == "Sigmoid":
            activation_output = edsl.sigmoid(z)
        # There is a bug in edsl.shape
        # elif op == "Relu":
        #     z_shape = edsl.shape(edsl.cast(z, dtype=edsl.fixed(14, 23)))
        #     ones = edsl.ones(z_shape, dtype=edsl.float64)
        #     ones = edsl.cast(ones, dtype=predictor_utils.DEFAULT_FIXED_DTYPE)
        #     zeros = edsl.sub(ones, ones)
        #     activation_output = edsl.maximum([zeros, z])
        elif activation == "Softmax":
            n_classes = np.shape(self.biases[-1])[0]
            activation_output = edsl.softmax(
                z,
                axis=1,
                upmost_index=n_classes,  # infer number of classes or regression targets
            )
        elif activation == "Identity":
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


class MultiLayerPerceptron(NeuralNetwork):
    @abc.abstractmethod
    def post_transform(self, y, fixedpoint_dtype):
        pass

    @classmethod
    def from_onnx(cls, model_proto):
        weights_data = predictor_utils.find_parameters_in_model_proto(
            model_proto, "coefficient", enforce=False
        )
        biases_data = predictor_utils.find_parameters_in_model_proto(
            model_proto, "intercepts", enforce=False
        )
        weights = []
        for weight in weights_data:
            dimentions = weight.dims
            assert weight is not None
            if weight.data_type != 1:  # FLOATS
                raise ValueError(
                    "MLP coefficients must be of type FLOATS, found other."
                )
            weight = np.asarray(weight.float_data)
            weight = weight.reshape(dimentions).T
            weights.append(weight)
        biases = []
        for bias in biases_data:
            assert bias is not None
            if bias.data_type != 1:  # FLOATS
                raise ValueError(
                    "MLP coefficients must be of type FLOATS, found other."
                )
            bias = np.asarray(bias.float_data)
            biases.append(bias)

        # parse activation function
        activation = predictor_utils.find_activation_in_model_proto(
            model_proto, "next_activations", enforce=False
        )
        return cls(weights, biases, activation)

    def apply_layer(self, input, i, fixedpoint_dtype):
        w = self.fixedpoint_constant(
            self.weights[i].T, plc=self.mirrored, dtype=fixedpoint_dtype
        )
        b = self.fixedpoint_constant(
            self.biases[i], plc=self.mirrored, dtype=fixedpoint_dtype
        )
        y = edsl.dot(input, w)
        z = edsl.add(y, b)
        return z

    def activation_fn(self, z):
        if self.activations == "Sigmoid":
            activation_output = edsl.sigmoid(z)
            # There is a bug in edsl.shape
            # elif self.activation == "Relu":
            #     y_1_shape = edsl.slice(edsl.shape(x), begin=0, end=1)
            #     ones = edsl.ones(y_1_shape, dtype=edsl.float64)
            #     ones = edsl.cast(ones, dtype=fixedpoint_dtype)
            #     zeros = edsl.sub(ones, ones)
            #     activation_output = edsl.maximum([zeros, y_1])
        elif self.activations is None:
            activation_output = z  # identity activation
        else:
            raise ValueError("Invalid or unsupported activation function")
        return activation_output

    def neural_predictor_fn(self, x, fixedpoint_dtype):
        num_hidden_layers = len(self.weights) - 1  # infer number of layers
        for i in range(num_hidden_layers + 1):
            x = self.apply_layer(x, i, fixedpoint_dtype)
            if i < num_hidden_layers:
                x = self.activation_fn(x)
            else:
                x = x
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
                pred = self.post_transform(y, fixedpoint_dtype)
            return self.handle_output(pred, prediction_handler=self.bob)

        return predictor


class MLPRegressor(MultiLayerPerceptron):
    def post_transform(self, y, fixedpoint_dtype):
        # no-op for linear regression models
        return y


class MLPClassifier(MultiLayerPerceptron):
    def post_transform(self, y, fixedpoint_dtype):
        # infer post_transform
        n_classes = np.shape(self.biases[-1])[0]
        print("n_classes ", n_classes)
        if n_classes == 1:
            self._post_transform = lambda x: self._sigmoid(
                x, predictor_utils.DEFAULT_FIXED_DTYPE
            )
            return self._post_transform(y)
        elif n_classes > 1:
            self._post_transform = lambda x: edsl.softmax(
                x, axis=1, upmost_index=n_classes
            )
            return self._post_transform(y)
        else:
            raise ValueError("Specify number of classes")

    def _sigmoid(self, y, fixedpoint_dtype):
        """
        returns both probabilities
        """
        pos_prob = edsl.sigmoid(y)
        one = self.fixedpoint_constant(1, plc=self.mirrored, dtype=fixedpoint_dtype)
        neg_prob = edsl.sub(one, pos_prob)
        return edsl.concatenate([neg_prob, pos_prob], axis=1)
