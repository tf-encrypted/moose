import abc

import numpy as np

from pymoose import edsl
from pymoose.predictors import aes_predictor
from pymoose.predictors import predictor_utils


class NeuralNetwork(aes_predictor.AesPredictor, metaclass=abc.ABCMeta):
    def __init__(self, weights_layer_1, biases_layer_1, weights_layer_2, biases_layer_2):
        super().__init__()
        self.weights_layer_1 = weights_layer_1
        self.biases_layer_1 = biases_layer_1
        self.weights_layer_2 = weights_layer_2
        self.biases_layer_2 = biases_layer_2

    @classmethod
    @abc.abstractmethod
    def from_onnx(cls, model_proto):
        pass

    @abc.abstractmethod
    def post_transform(self, y):
        pass

    @classmethod
    def bias_trick(cls, x, plc, dtype):
        bias_shape = edsl.slice(
            edsl.shape(x, placement=plc), begin=0, end=1, placement=plc
        )
        bias = edsl.ones(bias_shape, dtype=edsl.float64, placement=plc)
        reshaped_bias = edsl.expand_dims(bias, 1, placement=plc)
        return edsl.cast(reshaped_bias, dtype=dtype, placement=plc)

    def neural_predictor_fn(self, x, fixedpoint_dtype):
        # layer 1
        w_1 = self.fixedpoint_constant(
            np.concatenate([self.biases_layer_1.T, self.weights_layer_1], axis=1).T,
            plc=self.mirrored,
            dtype=fixedpoint_dtype,
        )
        bias_1 = self.bias_trick(x, plc=self.bob, dtype=fixedpoint_dtype)
        x_1 = edsl.concatenate([bias_1, x], axis=1)
        y_1 = edsl.dot(x_1, w_1)

        # relu
        # y_1 = edsl.cast(y_1, dtype=edsl.fixed(14, 23))
        # y_1_shape = edsl.shape(y_1)
        y_1_shape = edsl.shape(y_1, edsl.cast(y_1, dtype=edsl.fixed(14, 23)))
        ones = edsl.ones(y_1_shape, dtype=edsl.float64)
        ones = edsl.cast(ones, dtype=fixedpoint_dtype)
        zeros = edsl.sub(ones, ones)
        # zeros = edsl.cast(zeros, dtype=edsl.fixed(14, 23))
        relu_output = edsl.maximum([zeros, y_1])

        # layer 2
        w_2 = self.fixedpoint_constant(
        np.concatenate([self.biases_layer_2.T, self.weights_layer_2], axis=1).T,
        plc=self.mirrored,
        dtype=fixedpoint_dtype,
        )
        bias_2 = self.bias_trick(relu_output, plc=self.bob, dtype=fixedpoint_dtype)
        # bias_2 = edsl.cast(bias_2, dtype=edsl.fixed(14, 23))
        # relu_output = edsl.cast(relu_output, dtype=edsl.fixed(14, 23))
        x_2 = edsl.concatenate([bias_2, relu_output], axis=1)
        # x_2 = edsl.cast(x_2, dtype=edsl.fixed(14, 23))
        # x_2 = edsl.cast(x_2, dtype=edsl.fixed(14, 23))
        y_2 = edsl.dot(x_2, w_2)
        return y_2


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
                pred = self.post_transform(y)
            return self.handle_output(pred, prediction_handler=self.bob)

        return predictor


class NeuralRegressor(NeuralNetwork):
    def post_transform(self, y):
        # no-op for linear regression models
        return y

    @classmethod
    def from_onnx(cls, model_proto):
        lr_node = predictor_utils.find_node_in_model_proto(
            model_proto, "LinearRegressor", enforce=False
        )
        if lr_node is None:
            raise ValueError(
                "Incompatible ONNX graph provided: graph must contain a "
                "LinearRegressor operator."
            )

        coeffs_attr = predictor_utils.find_attribute_in_node(lr_node, "coefficients")
        if coeffs_attr.type != 6:  # FLOATS
            raise ValueError(
                "LinearRegressor coefficients must be of type FLOATS, found other."
            )
        coeffs = np.asarray(coeffs_attr.floats)
        # extract intercept if it's there, otherwise pass it as None
        intercepts_attr = predictor_utils.find_attribute_in_node(
            lr_node, "intercepts", enforce=False
        )
        if intercepts_attr is None:
            intercepts = None
        elif intercepts_attr.type != 6:  # FLOATS
            raise ValueError(
                "LinearRegressor intercept must be of type FLOATS, found other."
            )
        else:
            intercepts = intercepts_attr.floats

        # if n_targets is not None reshape into (n_targets, n_features) matrix
        n_targets_ints = predictor_utils.find_attribute_in_node(
            lr_node, "targets", enforce=False
        )
        if n_targets_ints is not None:
            n_targets = n_targets_ints.i
            coeffs = coeffs.reshape(n_targets, -1)

        return cls(coeffs=coeffs, intercepts=intercepts)


class NeuralClassifier(NeuralNetwork):
    def __init__(self, weights_layer_1, biases_layer_1, weights_layer_2, biases_layer_2, n_classes, transform_output=True):
        super().__init__(weights_layer_1, biases_layer_1, weights_layer_2, biases_layer_2)
        n_classes = n_classes

        # infer post_transform
        if n_classes == 2:
            self._post_transform = lambda x: edsl.sigmoid(x)
        elif n_classes > 2:
            self._post_transform = lambda x: edsl.softmax(
                x, axis=1, upmost_index=n_classes
            )
        else:
            raise ValueError("Specify number of classes")

    @classmethod
    def from_onnx(cls, model_proto):
        # parse classifier coefficients - weights of layer 1
        weight_1, dim_1 = predictor_utils.find_initializer_in_model_proto(model_proto, "coefficient", enforce=False)
        assert weight_1 is not None
        if weight_1.data_type != 1:  # FLOATS
            raise ValueError(
                "MLP coefficients must be of type FLOATS, found other."
            )
        # print(weight_1)
        weight_1 = np.asarray(weight_1.float_data)
        print(dim_1)
        print(weight_1.shape)
        weight_1 = weight_1.reshape(dim_1[1], -1)
        print(weight_1.shape)
        # parse classifier coefficients - weights of layer 2
        weight_2, dim_2 = predictor_utils.find_initializer_in_model_proto(model_proto, "coefficient1", enforce=False)
        assert weight_2 is not None
        if weight_2.data_type != 1:  # FLOATS
            raise ValueError(
                "MLP coefficients must be of type FLOATS, found other."
            )
        weight_2 = np.asarray(weight_2.float_data)
        weight_2 = weight_2.reshape(dim_2[1], -1)
        print(weight_2.shape)

        # labels
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

        # parse classifier biases of layer 1
        bias_1, dim_1 = predictor_utils.find_initializer_in_model_proto(model_proto, "intercepts", enforce=False)
        assert bias_1 is not None
        if bias_1.data_type != 1:  # FLOATS
            raise ValueError(
                "MLP coefficients must be of type FLOATS, found other."
            )
        bias_1 = np.asarray(bias_1.float_data)
        bias_1 = bias_1.reshape(dim_1[0], -1)
        
        # parse classifier biases of layer 2
        bias_2, dim_2 = predictor_utils.find_initializer_in_model_proto(model_proto, "intercepts1", enforce=False)
        assert bias_2 is not None
        if bias_2.data_type != 1:  # FLOATS
            raise ValueError(
                "MLP coefficients must be of type FLOATS, found other."
            )
        bias_2 = np.asarray(bias_2.float_data)
        bias_2 = bias_2.reshape(dim_2[0], -1)

        return cls(
            weight_1,
            bias_1,
            weight_2,
            bias_2,
            n_classes,
        )

    def post_transform(self, y):
        return self._post_transform(y)

    def _normalized_sigmoid(self, x, axis):
        y = edsl.sigmoid(x)
        return edsl.div(y, edsl.sum(y, axis))


def _validate_model_args(coeffs, intercepts):
    coeffs = _interpret_coeffs(coeffs)
    intercepts = _interpret_intercepts(intercepts)
    if intercepts is not None and coeffs.shape[0] != intercepts.shape[-1]:
        raise ValueError(
            "Shape mismatch between model coefficients and intercepts: "
            f"Intercepts size of {coeffs.shape[0]} inferred from coefficients, "
            f"found {intercepts.shape[-1]}."
        )
    return coeffs, intercepts


def _interpret_coeffs(coeffs):
    coeffs = np.asarray(coeffs, dtype=np.float64)
    coeffs_shape = coeffs.shape
    if len(coeffs_shape) == 1:
        return np.expand_dims(coeffs, 0)
    elif len(coeffs_shape) == 2:
        return coeffs
    raise ValueError(
        f"Coeffs must be convertible to a rank-2 tensor, found shape of {coeffs_shape}."
    )


def _interpret_intercepts(intercepts):
    if intercepts is None:
        return intercepts
    intercepts = np.asarray(intercepts, dtype=np.float64)
    intercepts_shape = intercepts.shape
    if len(intercepts_shape) == 1:
        return np.expand_dims(intercepts, 0)
    elif len(intercepts_shape) == 2:
        if intercepts_shape[0] != 1:
            pass
        else:
            return intercepts
    raise ValueError(
        f"Intercept must be convertible to a vector, found shape of {intercepts_shape}."
    )
