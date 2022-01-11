import abc

from pymoose import edsl

from . import model_utils as utils


class AesPredictor(metaclass=abc.ABCMeta):
    def __init__(self):
        (
            (self.alice, self.bob, self.carole),
            self.replicated,
        ) = self._standard_replicated_placements()

    def _standard_replicated_placements(self):
        alice = edsl.host_placement("alice")
        bob = edsl.host_placement("bob")
        carole = edsl.host_placement("carole")
        replicated = edsl.replicated_placement(
            name="replicated", players=[alice, bob, carole]
        )
        return (alice, bob, carole), replicated

    @property
    def host_placements(self):
        return self.alice, self.bob, self.carole

    @classmethod
    def fixedpoint_constant(cls, x, plc, dtype=utils.DEFAULT_FIXED_DTYPE):
        x = edsl.constant(x, dtype=edsl.float64, placement=plc)
        return edsl.cast(x, dtype=dtype, placement=plc)

    @abc.abstractmethod
    def predictor_factory(self, *args, **kwargs):
        pass
