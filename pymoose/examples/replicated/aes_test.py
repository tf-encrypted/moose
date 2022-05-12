import argparse
import logging

import numpy as np
import pytest
from absl.testing import absltest
from absl.testing import parameterized

import pymoose as pm
from pymoose.computation import utils
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class ReplicatedExample(parameterized.TestCase):
    def _setup_aes_comp(self, host_decrypt):
        alice = pm.host_placement(name="alice")
        bob = pm.host_placement(name="bob")
        carole = pm.host_placement(name="carole")
        rep = pm.replicated_placement(name="rep", players=[alice, bob, carole])

        decryptor = alice if host_decrypt else rep

        @pm.computation
        def my_aes_comp(
            key: pm.Argument(rep, vtype=pm.AesKeyType()),
            ciphertext: pm.Argument(
                alice, vtype=pm.AesTensorType(pm.fixed(24, 40))
            ),
        ):
            with decryptor:
                data = pm.decrypt(key, ciphertext)

            with alice:
                res = pm.cast(data, pm.float64)

            return res

        return my_aes_comp

    @parameterized.parameters(True, False)
    def test_aes_example_serde(self, host_decrypt):
        aes_comp = self._setup_aes_comp(host_decrypt)
        traced_aes_comp = pm.trace(aes_comp)
        comp_bin = utils.serialize_computation(traced_aes_comp)
        # Compile in Rust
        # If this does not error, rust was able to deserialize the pycomputation
        pm.elk_compiler.compile_computation(comp_bin, [])

    @parameterized.parameters(True, False)
    def test_aes_example_compile(self, host_decrypt):
        aes_comp = self._setup_aes_comp(host_decrypt)
        traced_aes_comp = pm.trace(aes_comp)
        comp_bin = utils.serialize_computation(traced_aes_comp)
        _ = pm.elk_compiler.compile_computation(comp_bin)

    @parameterized.parameters(True, False)
    @pytest.mark.slow
    def test_aes_example_execute(self, host_decrypt):
        aes_comp = self._setup_aes_comp(host_decrypt)
        traced_aes_comp = pm.trace(aes_comp)
        storage = {
            "alice": {},
            "bob": {},
            "carole": {},
        }
        runtime = LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=traced_aes_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={
                "key/alice/share0": np.array([0] * 128, dtype=np.bool_),
                "key/alice/share1": np.array([0] * 128, dtype=np.bool_),
                "key/bob/share1": np.array([0] * 128, dtype=np.bool_),
                "key/bob/share2": np.array([0] * 128, dtype=np.bool_),
                "key/carole/share2": np.array([0] * 128, dtype=np.bool_),
                "key/carole/share0": np.array([0] * 128, dtype=np.bool_),
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
