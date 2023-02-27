# Moose

[![crate][crate-image]][crate-link]
[![Docs][docs-image]][docs-link]
[![Build Status][build-image]][build-link]
[![Apache2 License 2.0][license-image]][license-link]
[![Minimum rustc version][rustc-image]][rustc-link]
[![Downloads][downloads-image]][crate-link]

Moose is a secure distributed dataflow framework consisting of a compiler, runtime, and Python eDSL and bindings. It is suitable for, but not limited to, encrypted machine learning and data processing. It's production ready and written primarily in Rust.

Computations are expressed using either the eDSL or by programming against the Rust API. Each operation in the dataflow graphs are pinned to a placement which represents either a physical machine or one of several kinds of virtual execution units. Moose currently includes support for simpler machine learning models, and a virtual placement backed by secure multi-party computation (MPC) in the form of replicated secret sharing. Please see [docs.rs](https://docs.rs/moose/), [examples](https://github.com/tf-encrypted/moose/tree/main/pymoose/examples), [tutorials](https://github.com/tf-encrypted/moose/tree/main/tutorials), or our [whitepaper](https://github.com/tf-encrypted/moose-whitepaper) for more details.

Moose is a community driven open source project and contributions are more than welcome. Moose was created at Cape.

## Example

The following is a simple example using the Python eDSL and bindings to express and evaluate an encrypted computation using replicated secret sharing.

```python
import numpy as np
import pymoose as pm

alice = pm.host_placement("alice")
bob = pm.host_placement("bob")
carole = pm.host_placement("carole")
replicated = pm.replicated_placement("rep", players=[alice, bob, carole])

@pm.computation
def simple_computation(
    x: pm.Argument(placement=alice, vtype=pm.TensorType(pm.float64)),
    y: pm.Argument(placement=bob, vtype=pm.TensorType(pm.float64)),
):
    with alice:
        x = pm.cast(x, dtype=pm.fixed(14, 23))

    with bob:
        y = pm.cast(y, dtype=pm.fixed(14, 23))

    with replicated:
        z = pm.add(x, y)

    with carole:
        v = pm.cast(z, dtype=pm.float64)

    return v

runtime = pm.GrpcMooseRuntime({
    alice: "localhost:50000",
    bob: "localhost:50001",
    carole: "localhost:50002",
})
runtime.set_default()

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


## Installation

Moose is packaged in two ways: the Python eDSL and bindings, and the CLI tools. In a typical use case you might want to install the Python bindings on your laptop and the CLI tools on the servers running in the distributed cluster (or use the [Docker image](https://hub.docker.com/r/tfencrypted/moose)).

Install the Python eDSL and bindings using:

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

Alternatively, you can install from the source code as described in [DEVELOP.md](https://github.com/tf-encrypted/moose/tree/main/DEVELOP.md).

## License

Moose is distributed under the terms of Apache License (Version 2.0). Copyright as specified in [NOTICE](https://github.com/tf-encrypted/moose/tree/main/NOTICE).


[//]: # (badges)


[crate-image]: https://img.shields.io/crates/v/moose.svg
[crate-link]: https://crates.io/crates/moose
[docs-image]: https://docs.rs/moose/badge.svg
[docs-link]: https://docs.rs/moose
[build-image]: https://github.com/tf-encrypted/moose/workflows/CI/badge.svg
[build-link]: https://github.com/tf-encrypted/moose/actions
[license-image]: https://img.shields.io/badge/license-Apache%20License%202.0-blue.svg?style=flat
[license-link]: https://www.apache.org/licenses/LICENSE-2.0
[rustc-image]: https://img.shields.io/badge/rustc-1.56+-blue.svg
[rustc-link]: https://github.com/tf-encrypted/moose#rust-version-requirements
[downloads-image]: https://img.shields.io/crates/d/moose.svg
