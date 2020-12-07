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
        "#ffa4b6",
        "#f765a3",
        "#a155b9",
        "#3caea3",
    ]
    placement_colors = dict()

    def pick_color(placement):
        if placement not in placement_colors:
            all_placements = sorted(computation.placements.keys())
            for i, candidate in enumerate(all_placements):
                if candidate == placement:
                    color_index = i % len(color_scheme)
                    placement_colors[placement] = color_scheme[color_index]
        return placement_colors[placement]

    dot = Digraph()
    # add nodes for ops
    for op in computation.operations.values():
        placement = computation.placement(op.placement_name)
        op_type = type(op).__name__
        if op_type.endswith("Operation"):
            op_type = op_type[: -len("Operation")]
        placement_type = type(placement).__name__
        if placement_type.endswith("Placement"):
            placement_type = placement_type[: -len("Placement")]
        node_label = f"{op.name}: {op_type}\n" f"@{placement.name}: {placement_type}"
        dot.node(
            op.name, node_label, color=pick_color(op.placement_name), shape="rectangle",
        )
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
