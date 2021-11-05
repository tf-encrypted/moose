import argparse
import logging
import unittest

from pymoose import edsl
from pymoose import elk_compiler
from pymoose.computation import utils
from pymoose.computation.standard import AesKeyType
from pymoose.computation.standard import AesTensorType
from pymoose.logger import get_logger


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AES example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    unittest.main()
