import asyncio
import unittest

from absl.testing import parameterized

from moose.computation import primitives as primitives_dialect
from moose.computation import ring as ring_dialect
from moose.computation import standard as standard_dialect
from moose.computation.base import Computation
from moose.computation.host import HostPlacement
from moose.edsl.base import host_placement
from moose.edsl.tracer import trace
from moose.executor.executor import AsyncExecutor
from moose.runtime import TestRuntime as Runtime


def _create_test_players(number_of_players=2):
    return [host_placement(name=f"player_{i}") for i in range(number_of_players)]


def _run_computation(comp, players):
    runtime = Runtime()
    placement_instantiation = {player: player.name for player in players}
    concrete_comp = trace(comp)
    runtime.evaluate_computation(
        concrete_comp, placement_instantiation=placement_instantiation
    )
    return runtime.get_executor(players[-1].name).store


class PrimitivesKernelTest(parameterized.TestCase):
    def test_derive_seed(self):
        nonce = bytes("hello", "utf-8")
        key = bytes("abcdefghijklmnop", "utf-8")
        comp = Computation(operations={}, placements={})
        alice = comp.add_placement(HostPlacement(name="alice"))
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="key",
                placement_name=alice.name,
                inputs={},
                value=key,
                output_type=primitives_dialect.PRFKeyType(),
            )
        )
        comp.add_operation(
            primitives_dialect.DeriveSeedOperation(
                name="derived_seed",
                placement_name=alice.name,
                inputs={"key": "key"},
                nonce=nonce,
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"value": "derived_seed"},
                key="seed",
            )
        )
        executor = AsyncExecutor(networking=None)
        task = executor.run_computation(
            comp,
            placement_instantiation={alice: alice.name},
            placement=alice.name,
            session_id="0123456789",
        )
        asyncio.get_event_loop().run_until_complete(task)
        assert len(executor.store["seed"]) == 16

    def test_sample_ring(self):
        seed = bytes("abcdefghijklmnop", "utf-8")
        comp = Computation(operations={}, placements={})
        alice = comp.add_placement(HostPlacement(name="alice"))
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="seed",
                placement_name=alice.name,
                inputs={},
                value=seed,
                output_type=primitives_dialect.PRFKeyType(),
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="x_shape",
                placement_name=alice.name,
                inputs={},
                value=(2, 2),
                output_type=standard_dialect.ShapeType(),
            )
        )
        comp.add_operation(
            ring_dialect.RingSampleOperation(
                name="sampled",
                placement_name=alice.name,
                inputs={"shape": "x_shape", "seed": "seed"},
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"value": "sampled"},
                key="x_sampled",
            )
        )
        executor = AsyncExecutor(networking=None)
        task = executor.run_computation(
            comp,
            placement_instantiation={alice: alice.name},
            placement=alice.name,
            session_id="0123456789",
        )
        asyncio.get_event_loop().run_until_complete(task)
        x = executor.store["x_sampled"]
        assert x.shape == (2, 2)


if __name__ == "__main__":
    unittest.main()
