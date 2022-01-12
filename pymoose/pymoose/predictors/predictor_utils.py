from pymoose import edsl

DEFAULT_FLOAT_DTYPE = edsl.float64
DEFAULT_FIXED_DTYPE = edsl.fixed(24, 40)


def find_attribute_in_node(node, attribute_name, enforce=True):
    node_attr = None
    for attr in node.attribute:
        if attr.name == attribute_name:
            node_attr = attr
    if enforce and node_attr is None:
        raise ValueError(
            f"Node {node.name} does not contain attribute {attribute_name}."
        )
    return node_attr


def find_input_shape(input_node):
    return input_node.type.tensor_type.shape.dim


def find_node_in_model_proto(model_proto, operator_name, enforce=True):
    node = None
    for operator in model_proto.graph.node:
        if operator.name == operator_name:
            node = operator
    if enforce and node is None:
        raise ValueError(f"Model proto does not contain operator {operator_name}.")
    return node
