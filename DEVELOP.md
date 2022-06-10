# Develop

You will need a working [installation of Rust](https://www.rust-lang.org/learn/get-started) to compile and test this project. We use the [stable toolchain](https://rust-lang.github.io/rustup/concepts/channels.html) which can be set as the default using `rustup default stable`.

Install OpenBLAS development headers via `libopenblas-dev` for Ubuntu.

## Getting started

To install Moose from source:

```sh
git clone https://github.com/tf-encrypted/moose
cd moose
```

## Installing binaries

If you wish you can install the Moose binaries as follows:

```sh
cargo install --path moose
```

but for development purposes this is often not the best approach. Instead we suggest you use `cargo run --bin` and perhaps create the following aliases for the binaries you would otherwise obtain via `cargo install`:

```sh
alias elk="cargo run --bin elk --"
alias comet="cargo run --bin comet --"
alias cometctl="cargo run --bin cometctl --"
alias rudolph="cargo run --bin rudolph --"
```

## Compiling

```sh
cargo build
```

## Testing

There are three types of testing regimes which can be found in the Makefile:

```sh
make test
make test-ci
make test-long
```

When doing local development we suggest using `make test` command. The
`make-ci` command is used mostly for ci purposes and runs a smaller range of test cases. For
a more extensive test suite we recommend using `make test-long` command.

## Logging

```sh
export RUST_LOG="moose=debug"
```

## Documentation

To generate documentation provided by rust using the source files use:

```
cargo doc --no-deps --open
```

In order to fetch the latest documentation on the cryptographic protocols implemented in moose
check our [whitepaper](https://github.com/tf-encrypted/moose-whitepaper)!

## Linting

## Pull requests

To ensure your changes will pass our CI, it's wise to run the following commands before committing:

```sh
make ci-ready

# or, more verbosely:

make fmt
make lint
make test-ci
```

## Memory profiling

We have been using [heaptrack](https://github.com/KDE/heaptrack) to measure the memory consumptions of computations, but in order to do so it appears that you need to launch the binaries directly, and not via `cargo run`.

This can be done by first building them, eg `rudolph` in this case:

```sh
cargo build --release --bin rudolph
```

and then passing the produced executable to heaptrack:

```sh
heaptrack ./target/release/rudolph
```

## Telemetry

Moose also has basic support for Jaeger/OpenTelemetry and telemetry can be turned on in the reindeer via the `--telemetry` flag.

Jaeger may be launched in docker using:

```sh
docker run -p6831:6831/udp -p6832:6832/udp -p16686:16686 jaegertracing/all-in-one:latest
```

We encourage setting `RUST_LOG` to an appropriate value to limit the number of spans generated, in particular for large computations.

## Releasing

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

## Rust tools

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

## Tips and Tricks

### `cargo watch`

During development it can be useful to have `cargo watch` automatically re-launch eg reindeer on code changes. This can be achieved as follows, in this case for Rudolph:

```sh
cargo watch -c -x 'run --bin rudolph -- --identity "localhost:50000" --port 50000 --sessions ./examples' -i examples
```

Note that `-i examples` means workers are not re-launched when files in `./examples` are changed.

# TODO


### Bootstrapping

Install python development headers for your OS. (eg - `python3-dev` for Ubuntu, or `python38-devel` for OpenSUSE).

To install the library and all of its dependencies, run:

```sh
make install
```

This unwraps into two other targets, which are kept separate for purposes of caching in CI:

```sh
make pydep  # install dependencies
make pylib  # install runtime Python library
```

You will also need to compile protobuf files before running any examples that use gRPC, which you can do via:

```sh
make build
```

### Running locally for testing

```sh
python main.py --runtime test
```
