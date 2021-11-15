import argparse
import logging

import numpy as np
import pytest
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose import elk_compiler
from pymoose.computation import utils
from pymoose.computation.standard import AesKeyType
from pymoose.computation.standard import AesTensorType
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class ReplicatedExample(parameterized.TestCase):
    def _setup_aes_comp(self, host_decrypt):
        alice = edsl.host_placement(name="alice")
        bob = edsl.host_placement(name="bob")
        carole = edsl.host_placement(name="carole")
        rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

        decryptor = alice if host_decrypt else rep

        @edsl.computation
        def my_aes_comp(
            key: edsl.Argument(rep, vtype=AesKeyType()),
            ciphertext: edsl.Argument(alice, vtype=AesTensorType(edsl.fixed(46, 40))),
        ):
            with decryptor:
                data = edsl.decrypt(key, ciphertext)

            with alice:
                res = edsl.cast(data, edsl.float64)

            return res

        return my_aes_comp

    @parameterized.parameters(True, False)
    def test_aes_example_serde(self, host_decrypt):
        aes_comp = self._setup_aes_comp(host_decrypt)
        traced_aes_comp = edsl.trace(aes_comp)
        comp_bin = utils.serialize_computation(traced_aes_comp)
        deser_aes_comp = utils.deserialize_computation(comp_bin)
        assert traced_aes_comp == deser_aes_comp

    @parameterized.parameters(True, False)
    def test_aes_example_rust_serde(self, host_decrypt):
        aes_comp = self._setup_aes_comp(host_decrypt)
        traced_aes_comp = edsl.trace(aes_comp)
        comp_bin = utils.serialize_computation(traced_aes_comp)
        # Compile in Rust
        # If this does not error, rust was able to deserialize the pycomputation
        elk_compiler.compile_computation(comp_bin, [])

    @parameterized.parameters(True, False)
    def test_aes_example_compile(self, host_decrypt):
        aes_comp = self._setup_aes_comp(host_decrypt)
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

    @parameterized.parameters(True, False)
    @pytest.mark.slow
    def test_aes_example_execute(self, host_decrypt):
        aes_comp = self._setup_aes_comp(host_decrypt)
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
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={
                "key/player0/share0": np.array([0] * 128, dtype=np.bool_),
                "key/player0/share1": np.array([0] * 128, dtype=np.bool_),
                "key/player1/share1": np.array([0] * 128, dtype=np.bool_),
                "key/player1/share2": np.array([0] * 128, dtype=np.bool_),
                "key/player2/share2": np.array([0] * 128, dtype=np.bool_),
                "key/player2/share0": np.array([0] * 128, dtype=np.bool_),
                "ciphertext": np.array([0] * 224, dtype=np.bool_),
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AES example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
