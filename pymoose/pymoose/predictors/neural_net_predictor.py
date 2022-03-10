import abc
from audioop import bias

import numpy as np

from pymoose import edsl
from pymoose.predictors import aes_predictor
from pymoose.predictors import predictor_utils
from pymoose.computation import standard as standard_ops

class NeuralNetwork(aes_predictor.AesPredictor, metaclass=abc.ABCMeta):
    def __init__(self, weights_layer_1, biases_layer_1, weights_layer_2, biases_layer_2, activation):
        super().__init__()
        self.weights_layer_1 = weights_layer_1
        self.biases_layer_1 = biases_layer_1
        self.weights_layer_2 = weights_layer_2
        self.biases_layer_2 = biases_layer_2
        self.activation = activation

    @classmethod
    @abc.abstractmethod
    def from_onnx(cls, model_proto):
        pass

    @abc.abstractmethod
    def post_transform(self, y):
        pass

    def neural_predictor_fn(self, x, fixedpoint_dtype):
        # layer 1
        w_1 = self.fixedpoint_constant(
            self.weights_layer_1.T,
            plc=self.mirrored,
            dtype=fixedpoint_dtype,
        )
        bias_1 = self.fixedpoint_constant(
            self.biases_layer_1,
            plc=self.mirrored,
            dtype=fixedpoint_dtype,
        )
        y_1 = edsl.dot(x, w_1)
        z_1 = edsl.add(y_1, bias_1)
        
        # activation function
        if self.activation == "Sigmoid":
            activation_output = edsl.sigmoid(z_1)
        # There is a bug in edsl.shape
        # elif self.activation == "Relu":
        #     y_1_shape = edsl.slice(edsl.shape(x), begin=0, end=1)
        #     ones = edsl.ones(y_1_shape, dtype=edsl.float64)
        #     ones = edsl.cast(ones, dtype=fixedpoint_dtype)
        #     zeros = edsl.sub(ones, ones)
        #     activation_output = edsl.maximum([zeros, y_1])
        else:
            activation_output = z_1 # no activation

        # layer 2
        w_2 = self.fixedpoint_constant(
            self.weights_layer_2.T,
            plc=self.mirrored,
            dtype=fixedpoint_dtype,
        )
        bias_2 = self.fixedpoint_constant(
            self.biases_layer_2,
            plc=self.mirrored,
            dtype=fixedpoint_dtype,
        )
        y_2 = edsl.dot(activation_output, w_2)
        z_2 = edsl.add(y_2, bias_2)
        return z_2


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
                activation = self.activation_function(x, fixedpoint_dtype)
                y = self.neural_predictor_fn(x, activation, fixedpoint_dtype)
                pred = self.post_transform(y)
            return self.handle_output(pred, prediction_handler=self.bob)

        return predictor


class NeuralRegressor(NeuralNetwork):
    def post_transform(self, y, fixedpoint_dtype):
        # no-op for linear regression models
        return y

    @classmethod
    def from_onnx(cls, model_proto):
        # parse classifier coefficients - weights of layer 1
        weight_1, dim_1 = predictor_utils.find_initializer_in_model_proto(model_proto, "coefficient", enforce=False)
        assert weight_1 is not None
        if weight_1.data_type != 1:  # FLOATS
            raise ValueError(
                "MLP coefficients must be of type FLOATS, found other."
            )
        weight_1 = np.asarray(weight_1.float_data)
        weight_1 = weight_1.reshape(dim_1).T

        # parse classifier coefficients - weights of layer 2
        weight_2, dim_2 = predictor_utils.find_initializer_in_model_proto(model_proto, "coefficient1", enforce=False)
        assert weight_2 is not None
        if weight_2.data_type != 1:  # FLOATS
            raise ValueError(
                "MLP coefficients must be of type FLOATS, found other."
            )
        weight_2 = np.asarray(weight_2.float_data)
        weight_2 = weight_2.reshape(dim_2).T

        # parse classifier biases of layer 1
        bias_1, dim_1 = predictor_utils.find_initializer_in_model_proto(model_proto, "intercepts", enforce=False)
        assert bias_1 is not None
        if bias_1.data_type != 1:  # FLOATS
            raise ValueError(
                "MLP coefficients must be of type FLOATS, found other."
            )
        bias_1 = np.asarray(bias_1.float_data)
        
        # parse classifier biases of layer 2
        bias_2, dim_2 = predictor_utils.find_initializer_in_model_proto(model_proto, "intercepts1", enforce=False)
        assert bias_2 is not None
        if bias_2.data_type != 1:  # FLOATS
            raise ValueError(
                "MLP coefficients must be of type FLOATS, found other."
            )
        bias_2 = np.asarray(bias_2.float_data)

        # parse activation function
        activation = predictor_utils.find_activation_in_model_proto(model_proto, "next_activations", enforce=False)

        # infer number of regression targets
        n_targets = dim_2[1]

        return cls(
            weight_1,
            bias_1,
            weight_2,
            bias_2,
            activation,
        )


class NeuralClassifier(NeuralNetwork):
    def __init__(self, weights_layer_1, biases_layer_1, weights_layer_2, biases_layer_2, n_classes, activation, transform_output=True):
        super().__init__(weights_layer_1, biases_layer_1, weights_layer_2, biases_layer_2, activation)
        n_classes = n_classes
        activation = activation

        # infer post_transform
        if n_classes == 2:
            self._post_transform = lambda x: self._sigmoid(x, predictor_utils.DEFAULT_FIXED_DTYPE)
        elif n_classes > 2:
            self._post_transform = lambda x: edsl.softmax(x, axis=1, upmost_index=n_classes)
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
        weight_1 = np.asarray(weight_1.float_data)
        weight_1 = weight_1.reshape(dim_1).T

        # parse classifier coefficients - weights of layer 2
        weight_2, dim_2 = predictor_utils.find_initializer_in_model_proto(model_proto, "coefficient1", enforce=False)
        assert weight_2 is not None
        if weight_2.data_type != 1:  # FLOATS
            raise ValueError(
                "MLP coefficients must be of type FLOATS, found other."
            )
        weight_2 = np.asarray(weight_2.float_data)
        weight_2 = weight_2.reshape(dim_2).T

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

        # parse classifier biases of layer 1
        bias_1, dim_1 = predictor_utils.find_initializer_in_model_proto(model_proto, "intercepts", enforce=False)
        assert bias_1 is not None
        if bias_1.data_type != 1:  # FLOATS
            raise ValueError(
                "MLP coefficients must be of type FLOATS, found other."
            )
        bias_1 = np.asarray(bias_1.float_data)
        
        # parse classifier biases of layer 2
        bias_2, dim_2 = predictor_utils.find_initializer_in_model_proto(model_proto, "intercepts1", enforce=False)
        assert bias_2 is not None
        if bias_2.data_type != 1:  # FLOATS
            raise ValueError(
                "MLP coefficients must be of type FLOATS, found other."
            )
        bias_2 = np.asarray(bias_2.float_data)

        # parse activation function
        activation = predictor_utils.find_activation_in_model_proto(model_proto, "next_activations", enforce=False)

        return cls(
            weight_1,
            bias_1,
            weight_2,
            bias_2,
            n_classes,
            activation,
        )

    def post_transform(self, y, fixedpoint_dtype):
        return self._post_transform(y)
    
    def _normalized_sigmoid(self, x, axis):
        y = edsl.sigmoid(x)
        return edsl.div(y, edsl.sum(y, axis))

    def _sigmoid(self, y, fixedpoint_dtype):
        '''
        returns both probabilities
        '''
        pos_prob = edsl.sigmoid(y)
        one = self.fixedpoint_constant(1, plc=self.mirrored, dtype=fixedpoint_dtype)
        neg_prob = edsl.sub(one, pos_prob)
        return edsl.concatenate([neg_prob, pos_prob], axis=1)