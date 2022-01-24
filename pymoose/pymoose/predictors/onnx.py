from pyexpat import model

from pymoose.predictors import linear_predictor
from pymoose.predictors import tree_ensemble


def from_onnx(model_proto):
    supported_op_type = [
        "LinearRegressor",
        "LinearClassifier",
        "TreeEnsembleRegressor",
        "TreeEnsembleClassifier",
    ]

    recognized_op = []
    unrocognized_op = []
    for node in model_proto.graph.node:
        node_type = node.op_type
        if node_type in supported_op_type:
            recognized_op.append(node_type)
        else:
            unrocognized_op.append(node_type)

    if len(recognized_op) == 1:
        model_type = recognized_op.pop()
    elif len(recognized_op) > 1:
        raise ValueError(
            "Incompatible ONNX graph provided: graph must contain at most "
            "one node of type LinearRegressor or LinearClassifier or "
            f"TreeEnsembleRegressor or TreeEnsembleClassifier, found {recognized_op}"
        )
    else:
        raise ValueError(
            "Incompatible ONNX graph provided: graph must contain a LinearRegressor "
            "or LinearClassifier or TreeEnsembleRegressor or TreeEnsembleClassifier node, "
            f"found: {unrocognized_op}"
        )

    if model_type == "LinearRegressor":
        model = linear_predictor.LinearRegressor.from_onnx(model_proto)
    elif model_type == "LinearClassifier":
        model = linear_predictor.LinearClassifier.from_onnx(model_proto)
    elif model_type == "TreeEnsembleRegressor":
        model = tree_ensemble.TreeEnsembleRegressor.from_onnx(model_proto)
    elif model_type == "TreeEnsembleClassifier":
        model = tree_ensemble.TreeEnsembleClassifier.from_onnx(model_proto)

    return model
