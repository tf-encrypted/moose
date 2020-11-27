from collections import defaultdict

from moose.compiler import host
from moose.compiler import replicated
from moose.compiler.render import render_computation
from moose.computation.base import Computation


class Compiler:
    def __init__(self, passes=None):
        self.passes = passes or [
            host.ApplyFunctionPass(),
            replicated.ReplicatedPass(),
            host.NetworkingPass(),
        ]
        self.operations = []
        self.name_counters = defaultdict(int)
        self.known_operations = defaultdict(dict)

    def run_passes(self, computation: Computation, render=False) -> Computation:
        if render:
            render_computation(computation, "Logical")
        for compiler_pass in self.passes:
            computation = compiler_pass.run(computation, context=self)
            if render:
                render_computation(computation, f"{type(compiler_pass).__name__}")
        return computation

    def get_fresh_name(self, prefix):
        count = self.name_counters[prefix]
        self.name_counters[prefix] += 1
        return f"{prefix}_{count}"
