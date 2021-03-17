from moose import edsl
from moose.computation.utils import serialize_computation

alice = edsl.host_placement(name="alice")
bob = edsl.host_placement(name="bob")
carole = edsl.host_placement(name="carole")
dave = edsl.host_placement(name="dave")
rep = edsl.replicated_placement(name="rep", players=[alice, bob, carole])

def f():
    @edsl.computation
    def my_comp():
        with alice:
            x = edsl.load("x", dtype=edsl.float64)
        with bob:
            y = edsl.load("y", dtype=edsl.float64)
        with rep:
            z1 = edsl.mul(x, y)

        with dave:
            res_dave = edsl.save("res", z1)

        return res_dave

    concrete_comp = edsl.trace(my_comp)
    return serialize_computation(concrete_comp)

f()
