from collections import defaultdict

from moose.compiler.host import HostApplyFunctionPass
from moose.compiler.host import NetworkingPass
from moose.compiler.mpspdz import MpspdzApplyFunctionPass
from moose.compiler.render import render_computation
from moose.compiler.replicated import ReplicatedPass
from moose.computation.base import Computation


class Compiler:
    def __init__(self, passes=None):
        self.passes = passes or [
            MpspdzApplyFunctionPass(),
            HostApplyFunctionPass(),
            ReplicatedPass(),
            NetworkingPass(),
        ]
        self.name_counters = defaultdict(int)
        self.known_operations = defaultdict(dict)

    def run_passes(self, computation: Computation, render=False) -> Computation:
        if render:
            render_computation(computation, "pass-0-logical")
        for i, compiler_pass in enumerate(self.passes):
            computation, performed_changes = compiler_pass.run(computation, context=self)
            if render and performed_changes:
                render_computation(
                    computation, f"pass-{i+1}-{type(compiler_pass).__name__.lower()}"
                )
        return computation

    def get_fresh_name(self, prefix):
        count = self.name_counters[prefix]
        self.name_counters[prefix] += 1
        return f"{prefix}_{count}"
