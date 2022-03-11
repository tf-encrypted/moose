import abc
from enum import Enum

import numpy as np

from pymoose import edsl
from pymoose.predictors import aes_predictor
from pymoose.predictors import predictor_utils


class PostTransform(Enum):
    NONE = 1
    SIGMOID = 2
    SOFTMAX = 3


class LinearPredictor(aes_predictor.AesPredictor, metaclass=abc.ABCMeta):
    def __init__(self, coeffs, intercepts=None):
        super().__init__()
        self.coeffs, self.intercepts = _validate_model_args(coeffs, intercepts)

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

    def linear_predictor_fn(self, x, fixedpoint_dtype):
        if self.intercepts is not None:
            w = self.fixedpoint_constant(
                np.concatenate([self.intercepts.T, self.coeffs], axis=1).T,
                plc=self.mirrored,
                dtype=fixedpoint_dtype,
            )
            bias = self.bias_trick(x, plc=self.bob, dtype=fixedpoint_dtype)
        else:
            w = self.fixedpoint_constant(
                self.coeffs.T, plc=self.mirrored, dtype=fixedpoint_dtype
            )
        if self.intercepts is not None:
            x = edsl.concatenate([bias, x], axis=1)

        y = edsl.dot(x, w)
        return y

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
                y = self.linear_predictor_fn(x, fixedpoint_dtype)
                pred = self.post_transform(y)
            return self.handle_output(pred, prediction_handler=self.bob)

        return predictor


class LinearRegressor(LinearPredictor):
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


class LinearClassifier(LinearPredictor):
    def __init__(
        self, coeffs, intercepts=None, post_transform=None, transform_output=True
    ):
        super().__init__(coeffs, intercepts)
        n_classes = self.coeffs.shape[0]
        # infer post_transform
        if post_transform == post_transform.NONE:
            self._post_transform = lambda x: x
        elif post_transform == post_transform.SIGMOID and n_classes == 2:
            self._post_transform = lambda x: edsl.sigmoid(x)
        elif post_transform == post_transform.SIGMOID and n_classes > 2:
            self._post_transform = lambda x: self._normalized_sigmoid(x, axis=1)
        elif post_transform == post_transform.SOFTMAX:
            self._post_transform = lambda x: edsl.softmax(
                x, axis=1, upmost_index=n_classes
            )
        else:
            raise ValueError("Could not infer post-transform in LinearClassifier")

    @classmethod
    def from_onnx(cls, model_proto):
        # parse LinearClassifier node
        lc_node = predictor_utils.find_node_in_model_proto(
            model_proto, "LinearClassifier", enforce=False
        )
        if lc_node is None:
            raise ValueError(
                "Incompatible ONNX graph provided: graph must contain a "
                "LinearClassifier operator."
            )

        # parse classifier coefficients
        coeffs_attr = predictor_utils.find_attribute_in_node(
            lc_node, "coefficients", enforce=False
        )
        assert coeffs_attr is not None
        if coeffs_attr.type != 6:  # FLOATS
            raise ValueError(
                "LinearClassifier coefficients must be of type FLOATS, found other."
            )
        coeffs = np.asarray(coeffs_attr.floats)

        # reshape into (n_classes, n_features) matrix
        classlabels_ints = predictor_utils.find_attribute_in_node(
            lc_node, "classlabels_ints", enforce=False
        )
        classlabels_strings = predictor_utils.find_attribute_in_node(
            lc_node, "classlabels_strings", enforce=False
        )
        assert classlabels_ints is not None or classlabels_strings is not None
        if classlabels_ints is not None:
            classlabels = classlabels_ints.ints
        elif classlabels_strings is not None:
            classlabels = classlabels_strings.strings
        n_classes = len(classlabels)
        coeffs = coeffs.reshape(n_classes, -1)

        # parse classifier intercepts
        intercepts_attr = predictor_utils.find_attribute_in_node(
            lc_node, "intercepts", enforce=False
        )
        if intercepts_attr is None:
            intercepts = None
        elif intercepts_attr.type != 6:  # FLOATS
            raise ValueError(
                "LinearClassifier intercept must be of type FLOATS, found other."
            )
        else:
            intercepts = np.asarray(intercepts_attr.floats).reshape(1, n_classes)

        # infer multitask arg from multi_class attribute
        multi_class_int = predictor_utils.find_attribute_in_node(lc_node, "multi_class")
        assert multi_class_int.type == 2  # INT
        multi_class = bool(multi_class_int.i)
        multitask = not multi_class

        # derive transform_output
        multi_class_int = predictor_utils.find_attribute_in_node(lc_node, "multi_class")
        post_transform = predictor_utils.find_attribute_in_node(
            lc_node, "post_transform"
        )
        post_transform_str = post_transform.s.decode()

        # sanity check that post_transform conforms to our expectations
        if post_transform_str in ["SOFTMAX", "SOFTMAX_ZERO"] and multitask:
            raise RuntimeError(
                f"Invalid post_transform {post_transform_str} for multitask=True."
            )

        if post_transform_str == "NONE":
            post_transform = PostTransform.NONE
        elif post_transform_str == "LOGISTIC":
            post_transform = PostTransform.SIGMOID
        elif post_transform_str == "SOFTMAX":
            post_transform = PostTransform.SOFTMAX
        else:
            raise RuntimeError(
                f"{post_transform_str} post_transform is unsupported for "
                "LinearClassifier."
            )

        return cls(coeffs=coeffs, intercepts=intercepts, post_transform=post_transform,)

    def post_transform(self, y):
        return self._post_transform(y)

    def _normalized_sigmoid(self, x, axis):
        y = edsl.sigmoid(x)
        y_sum = edsl.expand_dims(edsl.sum(y, axis), axis)
        return edsl.div(y, y_sum)


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
