import argparse
import statistics
import sys
import time

import numpy as np

import pymoose as pm

alice = pm.host_placement("alice")
bob = pm.host_placement("bob")
carole = pm.host_placement("carole")
repl = pm.replicated_placement(name="rep", players=[alice, bob, carole])
mirr = pm.mirrored_placement(name="mirr", players=[alice, bob, carole])

parser = argparse.ArgumentParser(description="Logistic regression training benchmarks")

parser.add_argument(
    "--n_exp",
    dest="n_exp",
    type=int,
    default="3",
    help="number of experiments to compute statistics from",
)

parser.add_argument(
    "--batch_size",
    dest="batch_size",
    type=int,
    default="128",
    help="batch size",
)

parser.add_argument(
    "--n_iter",
    dest="n_iter",
    type=int,
    default="10",
    help="number of iterations",
)

args = parser.parse_args()
N_EXP = args.n_exp


N_FEATURES = 100
BATCH_SIZE = args.batch_size
N_INSTANCES = BATCH_SIZE * args.n_iter  # number of iterations
N_BATCHES = N_INSTANCES // BATCH_SIZE
LEARNING_RATE = 0.1
MOMENTUM = 0.9
FIXED_DTYPE = pm.fixed(24, 40)
N_EPOCHS = 1


class LogisticRegressor:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def __call__(self, x):
        logit = pm.dot(x, self.W) + self.b
        return pm.sigmoid(logit)

    def loss_grad(self, y, y_hat):
        dy = y_hat - y
        return dy

    def backward(self, dy, x, batch_size_inv):
        xT = pm.transpose(x)
        dW = pm.mul(pm.dot(xT, dy), batch_size_inv)
        db = pm.mul(pm.sum(dy, axis=0), batch_size_inv)
        return dW, db

    def update(self, weights):
        assert len(weights) == 2
        self.W, self.b = weights

    @property
    def weights(self):
        return self.W, self.b


class SGDMomentum:
    def __init__(self, learning_rate, momentum):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self._grad_cache = None

    def step(self, wb, wb_grad):
        W, b = wb
        dW, db = wb_grad

        deltaW = dW * self.learning_rate
        deltab = db * self.learning_rate

        if self._grad_cache is not None:
            deltaW_0, deltab_0 = self._grad_cache
            deltaW += deltaW_0 * self.momentum
            deltab += deltab_0 * self.momentum
        self._grad_cache = (deltaW, deltab)

        W -= deltaW
        b -= deltab
        return W, b

    def update(self, hparams):
        assert len(hparams) == 2
        self.learning_rate, self.momentum = hparams


def to_fixedpoint(*tensors, fixed_dtype=FIXED_DTYPE):
    return [pm.cast(tensor, dtype=fixed_dtype) for tensor in tensors]


def from_fixedpoint(*tensors, dtype=pm.float64):
    return [pm.cast(tensor, dtype=dtype) for tensor in tensors]


@pm.computation
def train(
    x: pm.Argument(alice, dtype=pm.float64),
    y: pm.Argument(alice, dtype=pm.float64),
    w_0: pm.Argument(bob, dtype=pm.float64),
    b_0: pm.Argument(bob, dtype=pm.float64),
):

    with alice:
        x, y = to_fixedpoint(x, y)
        x_batches = [
            x[i * BATCH_SIZE : (i + 1) * BATCH_SIZE, :] for i in range(N_BATCHES)
        ]
        y_batches = [
            y[i * BATCH_SIZE : (i + 1) * BATCH_SIZE, :] for i in range(N_BATCHES)
        ]

    with bob:
        model = LogisticRegressor(w_0, b_0)
        model.update(to_fixedpoint(*model.weights))
        learning_rate = pm.constant(LEARNING_RATE, dtype=pm.float64)
        momentum = pm.constant(MOMENTUM, dtype=pm.float64)
        learning_rate, momentum = to_fixedpoint(learning_rate, momentum)
        optimizer = SGDMomentum(learning_rate, momentum)

    with mirr:
        # NOTE: since `batch_size` is used in public-private division in model.backward,
        # we want to pin it to a mirrored placement to avoid a full replicated.
        # we also invert the constant to use mul instead of div
        batch_size_inv = pm.constant(1.0 / BATCH_SIZE, dtype=FIXED_DTYPE)

    with repl:
        # NOTE: only share the input data once, otherwise sharing happens twice in below loop
        x_batches = [pm.identity(xb) for xb in x_batches]
        for _ in range(N_EPOCHS):
            for xb, yb in zip(x_batches, y_batches):
                y_hat = model(xb)
                dy = model.loss_grad(yb, y_hat)
                dW, db = model.backward(dy, xb, batch_size_inv)
                weights = optimizer.step(model.weights, (dW, db))
                model.update(weights)

    with bob:
        W, b = from_fixedpoint(*model.weights)

    return W, b


def synthetic_data(rng, n_rows, n_cols):
    x = rng.standard_normal((n_rows, n_cols))
    y = rng.integers(2, size=(n_rows, 1)).astype(np.float64)
    return x, y


def init_weights(n_features, n_outputs):
    W = np.zeros((n_features, n_outputs), dtype=np.float64)
    b = np.zeros((1, n_outputs), dtype=np.float64)
    return W, b


if __name__ == "__main__":
    rng = np.random.default_rng()
    x, y = synthetic_data(rng, N_INSTANCES, N_FEATURES)
    w_0, b_0 = init_weights(N_FEATURES, 1)

    role_map = {
        alice: "localhost:50000",
        bob: "localhost:50001",
        carole: "localhost:50002",
    }

    moose_timings = list()
    python_timings = list()

    for runs in range(N_EXP):
        runtime = pm.GrpcMooseRuntime(role_map)
        runtime.set_default()

        time0 = time.time()
        result, timings = train(x, y, w_0, b_0)

        python_timing = time.time() - time0
        moose_timing = max(timings.values()) / 1_000_000

        moose_timings.append(moose_timing)
        python_timings.append(python_timing)

    print("MIN/MAX/VARIANCE/MEAN")
    print(
        f"{min(moose_timings):.3f}/{max(moose_timings):.3f}/{statistics.variance(moose_timings):.3f}/{statistics.mean(moose_timings):.3f}"
    )
