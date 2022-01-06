import pathlib

import numpy as np

from pymoose import edsl

from . import model
from . import model_utils


class LinearPredictor(model.AesPredictorModel):
    def __init__(self, coeffs, intercept=None):
        super().__init__()
        if intercept is not None:
            raise NotImplementedError(
                "Intercept term not yet implemented for LinearPredictor."
            )
        self.coeffs = self._interpret_coeffs(coeffs)

    @classmethod
    def from_onnx_proto(cls, model_proto):
        lr_node = model_utils.find_node_in_model_proto(model_proto, "LinearRegressor")
        if lr_node is None:
            raise ValueError(
                "Incompatible ONNX graph provided: graph must contain a LinearRegressor operator."
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

    def linear_predictor_fn(self, x, fixedpoint_dtype):
        # TODO: handle intercept terms in here
        with self.alice:
            w = edsl.constant(self.coeffs, dtype=fixedpoint_dtype)

        with self.replicated:
            return edsl.dot(x, w)

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
            return model_utils.handle_predictor_output(y, prediction_handler=self.bob)

        return predictor

    def _interpret_coeffs(self, coeffs):
        coeffs = np.asarray(coeffs, dtype=np.float64)
        coeffs_shape = coeffs.shape
        if len(coeffs_shape) == 1:
            return np.expand_dims(coeffs, -1)
        elif len(coeffs_shape) == 2:
            return coeffs
        raise ValueError(
            f"Coeffs must be interpretable as a rank-2 tensor, found shape of {coeffs.shape}."
        )
