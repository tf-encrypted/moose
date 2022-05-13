import abc

import pymoose as pm
from pymoose.predictors import predictor_utils as utils


class AesPredictor(metaclass=abc.ABCMeta):
    def __init__(self):
        (
            (self.alice, self.bob, self.carole),
            self.mirrored,
            self.replicated,
        ) = self._standard_replicated_placements()

    @classmethod
    def fixedpoint_constant(cls, x, plc=None, dtype=utils.DEFAULT_FIXED_DTYPE):
        x = pm.constant(x, dtype=pm.float64, placement=plc)
        return pm.cast(x, dtype=dtype, placement=plc)

    @classmethod
    def handle_aes_input(cls, aes_key, aes_data, decryptor):
        assert isinstance(aes_data.vtype, pm.AesTensorType)
        assert aes_data.vtype.dtype.is_fixedpoint
        assert isinstance(aes_key.vtype, pm.AesKeyType)

        with decryptor:
            aes_inputs = pm.decrypt(aes_key, aes_data)

        return aes_inputs

    @classmethod
    def handle_output(
        cls, prediction, prediction_handler, output_dtype=utils.DEFAULT_FLOAT_DTYPE
    ):
        with prediction_handler:
            result = pm.cast(prediction, dtype=output_dtype)

        return result

    @property
    def host_placements(self):
        return self.alice, self.bob, self.carole

    @abc.abstractmethod
    def predictor_factory(self, *args, **kwargs):
        pass

    def _standard_replicated_placements(self):
        alice = pm.host_placement("alice")
        bob = pm.host_placement("bob")
        carole = pm.host_placement("carole")
        replicated = pm.replicated_placement(
            name="replicated", players=[alice, bob, carole]
        )
        mirrored = pm.mirrored_placement(name="mirrored", players=[alice, bob, carole])
        return (alice, bob, carole), mirrored, replicated
