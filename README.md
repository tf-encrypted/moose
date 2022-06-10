# Moose

Moose is a framework for secure multi-party computation, written in Rust.

Moose includes an API, a compiler, and a runtime. It’s designed to be secure, fast, scalable, and extensible. Moose is production ready and secure by default.

- The `Moose eDSL` is an easy-to-use language for expressing programs of algebraic operations on `n`-dimensional arrays, which may contain inputs from one or more parties. Moose programs can be written in Python ([PyMoose](/pymoose)) or directly in Rust ([examples](/moose/examples)).
- The `Moose Compiler` builds, type checks, and optimizes the operations into a distributed data flow graph. We call this encrypted data flow.
- The `Moose Runtime` securely and efficiently executes the graph across a network of (potentially untrusted) compute clusters while protecting the secrecy of the inputs.

Moose contains the mathematical primitives to compose machine learning and deep learning models.

Moose is designed to support many different secure multi-party computation protocols. Initially, replicated secret sharing is supported. Contributions of additional protocols are welcome.


The implementation is documented on [docs.rs](https://docs.rs/moose/) and the cryptographic protocols are documented in our [whitepaper](https://github.com/tf-encrypted/moose-whitepaper).

Moose is a community driven, open source project. Moose was created at Cape.

## Installation

You will need to have [Rust](https://www.rust-lang.org/learn/get-started) and OpenBLAS installed. Once you have these you can install Moose using:

```sh
cargo install moose
```

OpenBLAS may be installed using:

- Debian/Ubuntu: `sudo apt install libopenblas-dev`

- macOS: `homebrew install openblas`

Please see [DEVELOP.md](./DEVELOP.md) for further instructions on how to install and work with the Moose source code.

## Example

TODO

## License

Moose is distributed under the terms of Apache License (Version 2.0). Copyright as specified in [NOTICE](./NOTICE).
