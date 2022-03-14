import abc

import numpy as np

from pymoose import edsl
from pymoose.predictors import aes_predictor
from pymoose.predictors import predictor_utils


class MLPPredictor(aes_predictor.AesPredictor, metaclass=abc.ABCMeta):
    def __init__(self, weights, biases, activation):
        super().__init__()
        self.weights = weights
        self.biases = biases
        self.activation = activation

    @classmethod
    @abc.abstractmethod
    def from_onnx(cls, model_proto):
        pass

    @abc.abstractmethod
    def post_transform(self, y, fixedpoint_dtype):
        pass

    def apply_layer(self, input, num_hidden_layers, i, fixedpoint_dtype):
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
        if self.activation == "Sigmoid":
            activation_output = edsl.sigmoid(z)
            # There is a bug in edsl.shape
            # elif self.activation == "Relu":
            #     y_1_shape = edsl.slice(edsl.shape(x), begin=0, end=1)
            #     ones = edsl.ones(y_1_shape, dtype=edsl.float64)
            #     ones = edsl.cast(ones, dtype=fixedpoint_dtype)
            #     zeros = edsl.sub(ones, ones)
            #     activation_output = edsl.maximum([zeros, y_1])
        elif self.activation is None:
            activation_output = z  # identity activation
        else:
            raise ValueError("Invalid or unsupported activation function")
        return activation_output

    def neural_predictor_fn(self, x, fixedpoint_dtype):
        num_hidden_layers = len(self.weights) - 1  # infer number of layers
        for i in range(num_hidden_layers + 1):
            x = self.apply_layer(x, num_hidden_layers, i, fixedpoint_dtype)
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


class MLPRegressor(MLPPredictor):
    def post_transform(self, y, fixedpoint_dtype):
        # no-op for linear regression models
        return y

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


class MLPClassifier(MLPPredictor):
    def __init__(
        self, weights, biases, n_classes, activation, transform_output=True,
    ):
        super().__init__(weights, biases, activation)
        n_classes = n_classes
        activation = activation

        # infer post_transform
        if n_classes == 2:
            self._post_transform = lambda x: self._sigmoid(
                x, predictor_utils.DEFAULT_FIXED_DTYPE
            )
        elif n_classes > 2:
            self._post_transform = lambda x: edsl.softmax(
                x, axis=1, upmost_index=n_classes
            )
        else:
            raise ValueError("Specify number of classes")

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

        # parse labels
        labels_node = predictor_utils.find_node_in_model_proto(
            model_proto, "ZipMap", enforce=False
        )
        if labels_node is None:
            raise ValueError(
                "Incompatible ONNX graph provided: graph must contain a "
                "labels_node operator."
            )
        classlabels_ints = predictor_utils.find_attribute_in_node(
            labels_node, "classlabels_int64s", enforce=False
        )
        classlabels_strings = predictor_utils.find_attribute_in_node(
            labels_node, "classlabels_strings", enforce=False
        )
        assert classlabels_ints is not None or classlabels_strings is not None
        if classlabels_ints is not None:
            classlabels = classlabels_ints.ints
        elif classlabels_strings is not None:
            classlabels = classlabels_strings.strings
        n_classes = len(classlabels)

        # parse activation function
        activation = predictor_utils.find_activation_in_model_proto(
            model_proto, "next_activations", enforce=False
        )
        return cls(weights, biases, n_classes, activation)

    def post_transform(self, y, fixedpoint_dtype):
        return self._post_transform(y)

    def _sigmoid(self, y, fixedpoint_dtype):
        """
        returns both probabilities
        """
        pos_prob = edsl.sigmoid(y)
        one = self.fixedpoint_constant(1, plc=self.mirrored, dtype=fixedpoint_dtype)
        neg_prob = edsl.sub(one, pos_prob)
        return edsl.concatenate([neg_prob, pos_prob], axis=1)
