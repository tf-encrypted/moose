from graphviz import Digraph

from moose.computation.standard import ReceiveOperation
from moose.computation.standard import SendOperation


def render_computation(computation, filename_prefix="Physical", cleanup=True):
    color_scheme = [
        "#336699",
        "#ff0000",
        "#ff6600",
        "#92cd00",
        "#ffcc00",
    ]
    placement_colors = dict()

    def pick_color(placement):
        if placement not in placement_colors:
            color_index = len(placement_colors) % len(color_scheme)
            placement_colors[placement] = color_scheme[color_index]
        return placement_colors[placement]

    dot = Digraph()
    # add nodes for ops
    for op in computation.operations.values():
        op_type = type(op).__name__
        if op_type.endswith("Operation"):
            op_type = op_type[: -len("Operation")]
        dot.node(op.name, f"{op.name}: {op_type}", color=pick_color(op.placement_name))
    # add edges for explicit dependencies
    for op in computation.operations.values():
        for _, input_name in op.inputs.items():
            dot.edge(input_name, op.name)
    # add edges for implicit dependencies
    for recv_op in computation.operations.values():
        if not isinstance(recv_op, ReceiveOperation):
            continue
        for send_op in computation.operations.values():
            if not isinstance(send_op, SendOperation):
                continue
            if send_op.rendezvous_key == recv_op.rendezvous_key:
                dot.edge(
                    send_op.name,
                    recv_op.name,
                    label=send_op.rendezvous_key,
                    style="dotted",
                )
    dot.render(filename_prefix, format="png", cleanup=cleanup)
