import unittest

from absl.testing import parameterized

from pymoose.computation import ring as ring_dialect
from pymoose.computation import standard as standard_dialect
from pymoose.computation.base import Computation
from pymoose.computation.host import HostPlacement
from pymoose.computation.standard import BytesConstant
from pymoose.computation.standard import ShapeConstant
from pymoose.computation.standard import StringConstant
from pymoose.deprecated.computation import primitives as primitives_dialect
from pymoose.deprecated.testing import run_test_computation


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
                value=BytesConstant(value=key),
                output_type=primitives_dialect.PRFKeyType(),
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="save_key",
                placement_name=alice.name,
                inputs={},
                value=StringConstant(value="seed"),
                output_type=standard_dialect.StringType(),
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
                inputs={"key": "save_key", "value": "derived_seed"},
            )
        )

        results = run_test_computation(comp, [alice])
        assert len(results[alice]["seed"]) == 16

    def test_sample_ring(self):
        seed = bytes("abcdefghijklmnop", "utf-8")
        comp = Computation(operations={}, placements={})
        alice = comp.add_placement(HostPlacement(name="alice"))
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="seed",
                placement_name=alice.name,
                inputs={},
                value=BytesConstant(value=seed),
                output_type=primitives_dialect.PRFKeyType(),
            )
        )
        comp.add_operation(
            standard_dialect.ConstantOperation(
                name="x_shape",
                placement_name=alice.name,
                inputs={},
                value=ShapeConstant(value=(2, 2)),
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
            standard_dialect.ConstantOperation(
                name="save_key",
                placement_name=alice.name,
                inputs={},
                value=StringConstant(value="x_sampled"),
                output_type=standard_dialect.StringType(),
            )
        )
        comp.add_operation(
            standard_dialect.SaveOperation(
                name="save",
                placement_name=alice.name,
                inputs={"key": "save_key", "value": "sampled"},
            )
        )

        results = run_test_computation(comp, [alice])
        assert results[alice]["x_sampled"].shape == (2, 2)


if __name__ == "__main__":
    unittest.main()
