import argparse
import logging
import unittest
import numpy as np

from pymoose import edsl
from pymoose import elk_compiler
from pymoose.computation import utils
from pymoose.computation.standard import AesKeyType
from pymoose.computation.standard import AesTensorType
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class ReplicatedExample(unittest.TestCase):
    def _setup_aes_comp(self):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        @edsl.computation
        def my_aes_comp(
            key: edsl.Argument(rep, vtype=AesKeyType()),
            ciphertext: edsl.Argument(alice, vtype=AesTensorType(edsl.fixed(46, 40))),
        ):
            with rep:
                data = edsl.decrypt(key, ciphertext)

            with alice:
                res = edsl.cast(data, edsl.float64)

            return res

        return my_aes_comp

    def test_aes_example_serde(self):
        aes_comp = self._setup_aes_comp()
        traced_aes_comp = edsl.trace(aes_comp)
        comp_bin = utils.serialize_computation(traced_aes_comp)
        deser_aes_comp = utils.deserialize_computation(comp_bin)
        assert traced_aes_comp == deser_aes_comp

    def test_aes_example_rust_serde(self):
        aes_comp = self._setup_aes_comp()
        traced_aes_comp = edsl.trace(aes_comp)
        comp_bin = utils.serialize_computation(traced_aes_comp)
        # Compile in Rust
        # If this does not error, rust was able to deserialize the pycomputation
        elk_compiler.compile_computation(comp_bin, [])

    def test_aes_example_compile(self):
        aes_comp = self._setup_aes_comp()
        traced_aes_comp = edsl.trace(aes_comp)
        comp_bin = utils.serialize_computation(traced_aes_comp)
        _ = elk_compiler.compile_computation(
            comp_bin,
            [
                "typing",
                "full",
                # "print",
            ],
        )

    def test_aes_example_execute(self):
        aes_comp = self._setup_aes_comp()
        traced_aes_comp = edsl.trace(aes_comp)
        comp_bin = utils.serialize_computation(traced_aes_comp)
        compiled_comp = elk_compiler.compile_computation(
            comp_bin,
            [
                "typing",
                "full",
                "prune",
                "networking",
                # "print",
            ],
        )
        storage = {
            "alice": {},
            "bob": {},
            "carole": {},   
        }
        runtime = LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_compiled(
            comp_bin=compiled_comp,
            role_assignment={
                "alice": "alice",
                "bob": "bob",
                "carole": "carole",
            },
            arguments={
                "key/player0/share0": np.array([0] * 128, dtype=np.bool_),
                "key/player0/share1": np.array([0] * 128, dtype=np.bool_),
                "key/player1/share1": np.array([0] * 128, dtype=np.bool_),
                "key/player1/share2": np.array([0] * 128, dtype=np.bool_),
                "key/player2/share2": np.array([0] * 128, dtype=np.bool_),
                "key/player2/share0": np.array([0] * 128, dtype=np.bool_),
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AES example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    unittest.main()
