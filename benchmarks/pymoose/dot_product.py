import argparse
import statistics

import numpy as np

import pymoose as pm

alice = pm.host_placement("alice")
bob = pm.host_placement("bob")
carole = pm.host_placement("carole")

rep = pm.replicated_placement(name="rep", players=[alice, bob, carole])


def setup_par_dot_computation(n_parallel):
    @pm.computation
    def dot_product_comp(
        x_arg: pm.Argument(placement=alice, vtype=pm.TensorType(pm.float64)),
        y_arg: pm.Argument(placement=bob, vtype=pm.TensorType(pm.float64)),
    ):
        with alice:
            x = pm.cast(x_arg, dtype=pm.fixed(8, 27))

        with bob:
            y = pm.cast(y_arg, dtype=pm.fixed(8, 27))

        with rep:
            x_rep = pm.identity(x)
            y_rep = pm.identity(y)

            z_dots = [pm.dot(x_rep, y_rep) for _ in range(n_parallel)]

            z = pm.add_n(z_dots)

        with carole:
            res = pm.cast(z, pm.float64)

        return res

    return dot_product_comp


def setup_seq_dot_computation(n_seq):
    @pm.computation
    def dot_product_comp(
        x_arg: pm.Argument(placement=alice, vtype=pm.TensorType(pm.float64)),
        y_arg: pm.Argument(placement=bob, vtype=pm.TensorType(pm.float64)),
    ):
        with alice:
            x = pm.cast(x_arg, dtype=pm.fixed(8, 27))

        with bob:
            y = pm.cast(y_arg, dtype=pm.fixed(8, 27))

        with rep:
            y_rep = pm.identity(y)

            z_dots = [None] * n_seq
            z_dots[0] = pm.dot(x, y_rep)
            for i in range(1, n_seq):
                z_dots[i] = pm.dot(z_dots[i - 1], y_rep)

        with carole:
            res = pm.cast(z_dots[-1], pm.float64)

        return res

    return dot_product_comp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dot product benchmarks")

    parser.add_argument(
        "--c",
        dest="comp_type",
        type=str,
        default="parallel",
        help="computation type, seq or parallel",
    )

    parser.add_argument(
        "--s",
        dest="shape",
        type=int,
        default="1",
        help="shape used for dot products",
    )

    parser.add_argument(
        "--c_arg",
        dest="c_arg",
        type=int,
        default="1",
        help="number of dot products in parallel or sequence, depending on the computation type",
    )

    parser.add_argument(
        "--n",
        dest="n_iter",
        type=int,
        default="1",
        help="number of iterations for averaging the experiment",
    )

    args = parser.parse_args()
    comp_type = args.comp_type
    shape = args.shape
    n_iter = args.n_iter
    c_arg = args.c_arg

    role_map = {
        alice: "localhost:50000",
        bob: "localhost:50001",
        carole: "localhost:50002",
    }

    dot_product_comp = (
        setup_seq_dot_computation(c_arg)
        if comp_type == "seq"
        else setup_par_dot_computation(c_arg)
    )

    x = np.ones([shape, shape], dtype=np.float64)
    y = np.identity(shape, dtype=np.float64)

    AVG_TIME = 0

    moose_timings = list()

    for _ in range(n_iter):
        runtime = pm.GrpcMooseRuntime(role_map)
        runtime.set_default()

        outputs, timings = runtime.evaluate_computation(
            computation=dot_product_comp, arguments={"x_arg": x, "y_arg": y}
        )
        moose_timings.append(max(timings.values()) / 1_000_000)

    print(
        f"Dot Product with shape {shape} on {comp_type}-{c_arg} has all outputs ready in {statistics.mean(moose_timings)} ms on average"
    )
