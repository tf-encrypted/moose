import abc

import numpy as np
from pymoose import edsl

from . import model
from . import model_utils


class LinearPredictor(model.AesPredictor, metaclass=abc.ABCMeta):
    def __init__(self, coeffs, intercept=None):
        super().__init__()
        self.coeffs, self.intercept = _validate_model_args(coeffs, intercept)

    @abc.abstractmethod
    def post_transform(self, y):
        pass

    @classmethod
    @abc.abstractmethod
    def from_onnx(cls, model_proto):
        pass

    def linear_predictor_fn(self, x, fixedpoint_dtype):
        with self.alice:
            w = edsl.constant(self.coeffs, dtype=fixedpoint_dtype)
            # TODO: use bias trick instead of explicit add op for intercept
            if self.intercept is not None:
                b = edsl.constant(self.intercept, dtype=fixedpoint_dtype)

        with self.replicated:
            y = edsl.dot(x, w)
            if self.intercept is not None:
                y = edsl.add(y, b)
            return y

    def predictor_factory(self, fixedpoint_dtype=model_utils.DEFAULT_FIXED_DTYPE):
        @edsl.computation
        def predictor(
            aes_data: edsl.Argument(
                self.alice, vtype=edsl.AesTensorType(dtype=fixedpoint_dtype)
            ),
            aes_key: edsl.Argument(self.replicated, vtype=edsl.AesKeyType()),
        ):
            x = model_utils.handle_aes_predictor_input(
                aes_key, aes_data, decryptor=self.replicated
            )
            y = self.linear_predictor_fn(x, fixedpoint_dtype)
            pred = self.post_transform(y)
            return model_utils.handle_predictor_output(pred, prediction_handler=self.bob)

        return predictor


class LinearRegressor(LinearPredictor):
    def post_transform(y):
        # no-op for linear regression models
        return y

    @classmethod
    def from_onnx(cls, model_proto):
        lr_node = model_utils.find_node_in_model_proto(model_proto, "LinearRegressor")
        if lr_node is None:
            raise ValueError(
                "Incompatible ONNX graph provided: graph must contain a "
                "LinearRegressor operator."
            )

        coeffs_attr = model_utils.find_attribute_in_node(lr_node, "coefficients")
        assert coeffs_attr is not None
        if coeffs_attr.type != 6:  # FLOATS
            raise ValueError(
                "LinearRegressor coefficients must be of type FLOATS, found other."
            )
        coeffs = coeffs_attr.floats
        # extract intercept if it's there, otherwise pass it as None
        intercept_attr = model_utils.find_attribute_in_node(lr_node, "intercepts")
        if intercept_attr is None:
            intercept = None
        elif intercept_attr.type != 6:  # FLOATS
            raise ValueError(
                "LinearRegressor intercept must of type FLOATS, found other."
            )
        else:
            intercept = intercept_attr.floats

        return cls(coeffs=coeffs, intercept=intercept)


class LinearClassifier(LinearPredictor):

    def __init__(self, coeffs, intercept=None, multitask=False):
        super().__init__(coeffs, intercept)
        n_classes = self.coeffs.shape[-1]
        if multitask or n_classes == 1:
            self.post_transform = edsl.sigmoid
        elif n_classes > 1:
            # self.post_transform = edsl.softmax
            raise NotImplementedError("Softmax classifier not yet implemented.")
        else:
            raise ValueError(
                f"Improper `multitask` argument, expected bool but found {type(multitask)}."
            )

    @classmethod
    def from_onnx(cls, model_proto):
        lc_node = model_utils.find_node_in_model_proto(model_proto, "LinearClassifier")
        if lc_node is None:
            raise ValueError(
                "Incompatible ONNX graph provided: graph must contain a "
                "LinearClassifier operator."
            )
        # TODO
        raise Exception()


def _validate_model_args(coeffs, intercept):
    coeffs = _interpret_coeffs(coeffs)
    intercept = _interpret_intercept(intercept)
    if intercept is not None and coeffs.shape[-1] != intercept.shape[-1]:
        raise ValueError(
            "Shape mismatch between model coefficients and intercept: "
            f"Intercept size of {coeffs.shape[-1]} inferred from coefficients, "
            f"found {intercept.shape[-1]}."
        )
    return coeffs, intercept


def _interpret_coeffs(coeffs):
    coeffs = np.asarray(coeffs, dtype=np.float64)
    coeffs_shape = coeffs.shape
    if len(coeffs_shape) == 1:
        return np.expand_dims(coeffs, -1)
    elif len(coeffs_shape) == 2:
        return coeffs
    raise ValueError(
        f"Coeffs must be convertible to a rank-2 tensor, found shape of {coeffs_shape}."
    )


def _interpret_intercept(intercept):
    if intercept is None:
        return intercept
    intercept = np.asarray(intercept, dtype=np.float64)
    intercept_shape = intercept.shape
    if len(intercept_shape) == 1:
        return np.expand_dims(intercept, 0)
    elif len(intercept_shape) == 2:
        if intercept_shape[0] != 1:
            pass
        else:
            return intercept
    raise ValueError(
        f"Intercept must be convertible to a vector, found shape of {intercept_shape}."
    )
