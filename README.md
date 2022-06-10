# Moose

Moose is a distributed dataflow framework for encrypted machine learning and data processing. It is written primarily in Rust and includes a compiler, runtime, and Python eDSL and bindings.

Computations are expressed using either the Python eDSL or by programming against the Rust API. These _logical computations_ are compiled to _physical computations_ that in turn can be executed using the runtime. Each operation in the dataflow graphs are pinned to a _placement_ which represent either a physical host or one of several kinds of virtual execution units, including some backed by secure multi-party computation (MPC) protocols.

Moose includes an API, a compiler, and a runtime. It’s designed to be secure, fast, scalable, and extensible. Moose is production ready and secure by default.

- The `Moose eDSL` is an easy-to-use language for expressing programs of algebraic operations on `n`-dimensional arrays, which may contain inputs from one or more parties. Moose programs can be written in Python ([PyMoose](/pymoose)) or directly in Rust ([examples](/moose/examples)).
- The `Moose Compiler` builds, type checks, and optimizes the operations into a distributed data flow graph. We call this encrypted data flow.
- The `Moose Runtime` securely and efficiently executes the graph across a network of (potentially untrusted) compute clusters while protecting the secrecy of the inputs.

Moose contains the mathematical primitives to compose machine learning and deep learning models.

Moose is designed to support many different secure multi-party computation protocols. Initially, replicated secret sharing is supported. Contributions of additional protocols are welcome.


The implementation is documented on [docs.rs](https://docs.rs/moose/) and the cryptographic protocols are documented in our [whitepaper](https://github.com/tf-encrypted/moose-whitepaper).

Moose is a community driven, open source project. Moose was created at Cape.

## Installation

Moose is packaged in two ways: the Python bindings and the CLI tools. In a typical use case you might want to install the Python bindings on your laptop and the CLI tools on the servers running in the distributed cluster (or use the [Docker image](https://hub.docker.com/r/tfencrypted/moose)).

Install the Python bindings using:

```sh
pip install moose-python
```

Install the CLI tools using (assuming you already have [Rust installed](https://www.rust-lang.org/learn/get-started)):

```sh
cargo install moose
```

You will also need to have OpenBLAS installed in both cases:

- Debian/Ubuntu: `sudo apt install libopenblas-dev`

- macOS: `homebrew install openblas`

Alternatively, you can install from the source code. Please see [DEVELOP.md](./DEVELOP.md) for further instructions on how to do that, as well as working with it.

## Example

The following is a simple example using the Python bindings to express and evaluate an encrypted computation using replicated secret sharing:

```python
import numpy as np
import pymoose as pm

alice = pm.host_placement("alice")
bob = pm.host_placement("bob")
carole = pm.host_placement("carole")
rep = pm.replicated_placement("rep", players=[alice, bob, carole])

runtime = pm.GrpcMooseRuntime({
    alice: "localhost:50000",
    bob: "localhost:50001",
    carole: "localhost:50002",
})
runtime.set_default()

@pm.computation
def simple_computation(
    x: pm.Argument(placement=alice, vtype=pm.TensorType(pm.float64)),
    y: pm.Argument(placement=bob, vtype=pm.TensorType(pm.float64)),
):
    with alice:
        x = pm.cast(x, dtype=pm.fixed(14, 23))

    with bob:
        y = pm.cast(y, dtype=pm.fixed(14, 23))

    with rep:
        z = pm.add(x, y)

    with carole:
        v = pm.cast(z, dtype=pm.float64)

    return v

result = my_computation(
    x=np.array([1.0, 2.0], dtype=np.float64),
    y=np.array([3.0, 4.0], dtype=np.float64),
)
print(result)
```

Make sure to have three instances of Comet running before running the Python code:

```sh
comet --identity localhost:50000 --port 50000
comet --identity localhost:50001 --port 50001
comet --identity localhost:50002 --port 50002
```

In this example the inputs are provided by the Python script but Moose also supports loading data locally from e.g. NumPy files.

Please see the [examples](./examples/) directory for more.

## License

Moose is distributed under the terms of Apache License (Version 2.0). Copyright as specified in [NOTICE](./NOTICE).
