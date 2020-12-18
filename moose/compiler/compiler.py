from collections import defaultdict

from moose.compiler.render import render_computation
from moose.computation.base import Computation


class Compiler:
    def __init__(self, passes):
        self.passes = passes
        self.name_counters = defaultdict(int)

    def run_passes(
        self,
        computation: Computation,
        render=False,
        render_edge_types=True,
        render_prefix="pass",
    ) -> Computation:
        if render:
            render_computation(
                computation,
                filename_prefix=f"{render_prefix}-0-initial",
                render_edge_types=render_edge_types,
            )
        for i, compiler_pass in enumerate(self.passes):
            computation, performed_changes = compiler_pass.run(
                computation, context=self
            )
            if render and performed_changes:
                render_computation(
                    computation,
                    filename_prefix=(
                        f"{render_prefix}-{i+1}"
                        f"-{type(compiler_pass).__name__.lower()}"
                    ),
                    render_edge_types=render_edge_types,
                )
        return computation

    def get_fresh_name(self, prefix):
        count = self.name_counters[prefix]
        self.name_counters[prefix] += 1
        return f"{prefix}_{count}"
