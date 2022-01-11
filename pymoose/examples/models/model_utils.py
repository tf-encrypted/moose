from pymoose import edsl

DEFAULT_FIXED_DTYPE = edsl.fixed(24, 40)


def find_attribute_in_node(node, attribute_name, enforce=False):
    node_attr = None
    for attr in node.attribute:
        if attr.name == attribute_name:
            node_attr = attr
    if enforce and node_attr is None:
        raise ValueError(f"Node {node.name} does not contain attribute {attr.name}.")
    return node_attr


def find_node_in_model_proto(model_proto, operator_name, enforce=False):
    node = None
    for operator in model_proto.graph.node:
        if operator.name == operator_name:
            node = operator
    if enforce and node is None:
        raise ValueError(f"Model proto does not contain operator {operator_name}.")
    return node


def handle_aes_predictor_input(aes_key, aes_data, decryptor):
    assert isinstance(aes_data.vtype, edsl.AesTensorType)
    assert aes_data.vtype.dtype.is_fixedpoint
    assert isinstance(aes_key.vtype, edsl.AesKeyType)

    with decryptor:
        aes_inputs = edsl.decrypt(aes_key, aes_data)

    return aes_inputs


def handle_predictor_output(prediction, prediction_handler, output_dtype=edsl.float64):
    with prediction_handler:
        result = edsl.cast(prediction, dtype=output_dtype)

    return result
