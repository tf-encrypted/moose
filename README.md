# Moose

Moose is a framework for secure multi-party computation, written in Rust.

Moose includes an API, a compiler, and a runtime. It’s designed to be secure, fast, scalable, and extensible. Moose is production ready and secure by default.

### Components

- The `Moose eDSL` is an easy-to-use language for expressing programs of algebraic operations on `n`-dimensional arrays, which may contain inputs from one or more parties. Moose programs can be written in Python ([PyMoose](/pymoose)) or directly in Rust ([examples](/moose/examples)).
- The `Moose Compiler` builds, type checks, and optimizes the operations into a distributed data flow graph. We call this encrypted data flow.
- The `Moose Runtime` securely and efficiently executes the graph across a network of (potentially untrusted) compute clusters while protecting the secrecy of the inputs.

Moose contains the mathematical primitives to compose machine learning and deep learning models.

Moose is designed to support many different secure multi-party computation protocols. Initially, replicated secret sharing is supported. Contributions of additional protocols are welcome.

### Community

Moose is a community driven, open source project. Moose was created at Cape.

## Installation

### Bootstrapping

**NOTE**: We are using [rust stable](https://rust-lang.github.io/rustup/concepts/channels.html) for the runtime.

Install python development headers for your OS. (eg - `python3-dev` for Ubuntu, or `python38-devel` for OpenSUSE).

Install OpenBLAS development headers via `libopenblas-dev` for Ubuntu.

To install the library and all of its dependencies, run:
```
make install
```

This unwraps into two other targets, which are kept separate for purposes of caching in CI:

```
make pydep  # install dependencies
make pylib  # install runtime Python library
```

You will also need to compile protobuf files before running any examples that use gRPC, which you can do via:

```
make build
```

### Running locally for testing

```
python main.py --runtime test
```

### Running with remote cluster

You can start a cluster locally using the following:

```
cd docker/dev
docker-compose up
```

Once done you can run the following to evaluate a computation on it:

```
python main.py --runtime remote
```

### Developing

To ensure your changes will pass our CI, it's wise to run the following commands before committing:

```
make ci-ready

# or, more verbosely:

make fmt
make lint
make test-ci
```

### Testing

There are three types of testing regimes which can be found in the Makefile:

```
make test
make test-ci
make test-long
```

When doing local development we suggest using `make test` command. The
`make-ci` command is used mostly for ci purposes and runs a smaller range of test cases. For
a more extensive test suite we recommend using `make test-long` command.

### Documentation

To generate documentation provided by rust using the source files use:

```
cargo doc --no-deps --open
```

In order to fetch the latest documentation on the cryptographic protocols implemented in moose
check our [whitepaper](https://github.com/tf-encrypted/moose-whitepaper)!

### Releasing

Follow these steps to release a new version:

0. Make sure `cargo release` is installed (`cargo install cargo-release`)

1. create a new branch from `main`, eg `git checkout -b new-release`

2. run `make release`

3. Update the [CHANGELOG.md](CHANGELOG.md) file to include notable changes since the last release.

4. if successful then `git push` to create a new PR

Once your PR has been merged to `main`:

1. checkout main branch: `git checkout main`

2. create a new tag *matching the version* of `python-bindings`: eg `git tag v{x.y.z}`

3. push tag: `git push origin v{x.y.z}`

4. create a release on GitHub based on your [tag](https://github.com/tf-encrypted/runtime/tags)

5. additionally tag the new versioned release with the `stable` tag, if the release is deemed stable.

6. update to the next dev version with `cargo release --workspace --no-publish beta --execute` and create a PR for that

If needed then tags on GitHub can be deleted using `git push --delete origin {tag-name}`

### Rust Development

You will need a working [installation of Rust](https://www.rust-lang.org/learn/get-started) to compile and test this project.

We compile and test against the stable toolchain so make sure to either set the stable toolchain as the default using `rustup default stable`.

We require code to be formatted according to `cargo fmt` so make sure to run this command before submitted your work. You should also run `cargo clippy` to lint your code.

To ease your development we encourage you to install the following extra cargo commands:

- [`cargo watch`](https://crates.io/crates/cargo-watchcargo-watch) will type check your code on every save;  `cargo watch --exec test` will run all tests on every save.

- [`cargo outdated`](https://crates.io/crates/cargo-outdated) checks if your dependencies are up to date.

- [`cargo audit`](https://crates.io/crates/cargo-audit) checks if any vulnerabilities have been detected for your current dependencies.

- [`cargo deny`](https://github.com/EmbarkStudios/cargo-deny) checks security advisories and licence conflicts.

- [`cargo release`](https://crates.io/crates/cargo-release) automates the release cycle, including bumping versions.

- [`cargo udeps`](https://crates.io/crates/cargo-udeps) to list unused dependencies.

- [`cargo expand`](https://github.com/dtolnay/cargo-expand) to dump what macros expand into.

- [`cargo asm`](https://github.com/gnzlbg/cargo-asm) to dump assembly or LLVM IR (the latter via `cargo llvm-ir`).

- [`cargo llvm-lines`](https://github.com/dtolnay/cargo-llvm-lines) to inspect code bloat.

[Tokio Console](https://tokio.rs/blog/2021-12-announcing-tokio-console) is also interesting.

To keep all of these up-to-date once installed, we recommend using [`cargo-update`](https://crates.io/crates/cargo-update).

## License

Moose is distributed under the terms of Apache License (Version 2.0). Copyright as specified in [NOTICE](./NOTICE).
