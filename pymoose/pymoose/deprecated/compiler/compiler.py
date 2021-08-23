from collections import defaultdict

from pymoose.computation.base import Computation
from pymoose.deprecated.compiler import host
from pymoose.deprecated.compiler import mpspdz
from pymoose.deprecated.compiler import pruning
from pymoose.deprecated.compiler import render as rendering
from pymoose.deprecated.compiler.fixedpoint import host_encoding_pass
from pymoose.deprecated.compiler.fixedpoint import host_lowering_pass
from pymoose.deprecated.compiler.fixedpoint import host_ring_lowering_pass
from pymoose.deprecated.compiler.replicated import encoding_pass
from pymoose.deprecated.compiler.replicated import lowering_pass
from pymoose.deprecated.compiler.replicated import replicated_pass
from pymoose.logger import get_logger


class Compiler:
    def __init__(self, passes=None, ring=64):
        self.passes = (
            passes
            if passes is not None
            else [
                mpspdz.MpspdzApplyFunctionPass(),
                host_encoding_pass.HostEncodingPass(),
                host_lowering_pass.HostLoweringPass(),
                encoding_pass.ReplicatedEncodingPass(),
                replicated_pass.ReplicatedOpsPass(),
                host_ring_lowering_pass.HostRingLoweringPass(),
                lowering_pass.ReplicatedLoweringPass(ring=ring),
                pruning.PruningPass(),
                host.NetworkingPass(),
            ]
        )
        self.name_counters = defaultdict(int)

    def run_passes(
        self,
        computation: Computation,
        render=False,
        render_edge_types=True,
        render_prefix="pass",
    ) -> Computation:
        if render:
            rendering.render_computation(
                computation,
                filename_prefix=f"{render_prefix}-0-initial",
                render_edge_types=render_edge_types,
            )
        for i, compiler_pass in enumerate(self.passes):
            computation, performed_changes = compiler_pass.run(
                computation, context=self
            )
            if render and performed_changes:
                rendering.render_computation(
                    computation,
                    filename_prefix=(
                        f"{render_prefix}-{i+1}"
                        f"-{type(compiler_pass).__name__.lower()}"
                    ),
                    render_edge_types=render_edge_types,
                )
        return computation

    def compile(
        self,
        computation: Computation,
        render=False,
        render_edge_types=True,
        render_prefix="pass",
    ) -> Computation:
        computation = self.run_passes(
            computation, render, render_edge_types, render_prefix,
        )
        for op in computation.operations.values():
            get_logger().debug(f"Computation: {op}")
        return computation

    def get_fresh_name(self, prefix):
        count = self.name_counters[prefix]
        self.name_counters[prefix] += 1
        return f"{prefix}_{count}"
