from pymoose.predictors import linear_predictor
from pymoose.predictors import neural_net_predictor
from pymoose.predictors import predictor_utils
from pymoose.predictors import tree_ensemble


def from_onnx(model_proto):
    supported_op_type = [
        "LinearRegressor",
        "LinearClassifier",
        "TreeEnsembleRegressor",
        "TreeEnsembleClassifier",
    ]

    recognized_ops = []
    unrocognized_ops = []
    for node in model_proto.graph.node:
        node_type = node.op_type
        if node_type in supported_op_type:
            recognized_ops.append(node_type)
        else:
            unrocognized_ops.append(node_type)

    if len(recognized_ops) == 1:
        model_type = recognized_ops.pop()

    elif len(recognized_ops) > 1:
        raise ValueError(
            "Incompatible ONNX graph provided: graph must contain at most "
            "one node of type LinearRegressor or LinearClassifier or "
            f"TreeEnsembleRegressor or TreeEnsembleClassifier, found {recognized_ops}"
        )
    # MultiLayerPerceptron (MLP) onnx graph does not contain a node name that identifies that the model is MLP
    # However, an MLP has at least two sets of weights
    elif (
        len(
            predictor_utils.find_parameters_in_model_proto(
                model_proto, "coefficient", enforce=False
            )
        )
        > 1
    ):
        model_type = "MLP"

    else:
        raise ValueError(
            "Incompatible ONNX graph provided: graph must contain a LinearRegressor "
            "or LinearClassifier or TreeEnsembleRegressor or TreeEnsembleClassifier "
            f"node, found: {unrocognized_ops}"
        )

    if model_type == "LinearRegressor":
        return linear_predictor.LinearRegressor.from_onnx(model_proto)
    elif model_type == "LinearClassifier":
        return linear_predictor.LinearClassifier.from_onnx(model_proto)
    elif model_type == "TreeEnsembleRegressor":
        return tree_ensemble.TreeEnsembleRegressor.from_onnx(model_proto)
    elif model_type == "TreeEnsembleClassifier":
        return tree_ensemble.TreeEnsembleClassifier.from_onnx(model_proto)
    elif (
        model_type == "MLP"
        and predictor_utils.find_node_in_model_proto(
            model_proto, "ZipMap", enforce=False
        )
        == None
    ):
        return neural_net_predictor.MLPRegressor.from_onnx(model_proto)
    elif (
        model_type == "MLP"
        and predictor_utils.find_node_in_model_proto(
            model_proto, "ZipMap", enforce=False
        )
        != None
    ):
        return neural_net_predictor.MLPClassifier.from_onnx(model_proto)
