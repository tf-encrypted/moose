# Develop

## Getting started

To install Moose from source:

```sh
git clone https://github.com/tf-encrypted/moose
cd moose
```

You will need a working [installation of Rust](https://www.rust-lang.org/learn/get-started) to compile and test this project; we generally use the [stable toolchain](https://rust-lang.github.io/rustup/concepts/channels.html).

### Debian/Ubuntu

Install dependencies:

```sh
sudo apt install libopenblas-dev
sudo apt install python3-dev
```

### macOS

Install dependencies (using [Homebrew](https://brew.sh/)):

```sh
brew install openblas
```

## Installing

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

```sh
cargo doc --no-deps --open
```

Our [whitepaper](https://github.com/tf-encrypted/moose-whitepaper) contains more documentation on the cryptographic protocols implemented in Moose.

## Formatting and linting

We require code to be formatted according to:

```sh
cargo fmt
```

so make sure to run this command before submitted your work. You should also run 

```sh
cargo clippy --all-targets --no-deps -- -D warnings
```

to lint your code.

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

### Moose

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

### PyMoose

TODO

### Docker

Build and push the Docker image using:

```sh
docker build -t tfencrypted/moose .
docker push tfencrypted/moose
```

## Rust tools

To ease your development we encourage you to install the following cargo subcommands:

- [`install-update`](https://crates.io/crates/cargo-update) to keep cargo subcommands up-to-date.

- [`watch`](https://crates.io/crates/cargo-watchcargo-watch) to type check your code on every save; `cargo watch --exec test` will run all tests on every save.

- [`outdated`](https://crates.io/crates/cargo-outdated) to check if your dependencies are up to date.

- [`audit`](https://crates.io/crates/cargo-audit) to check if any vulnerabilities have been detected for your current dependencies.

- [`deny`](https://github.com/EmbarkStudios/cargo-deny) to check for security advisories and license conflicts.

- [`release`](https://crates.io/crates/cargo-release) to automate the release cycle, including bumping versions.

- [`udeps`](https://crates.io/crates/cargo-udeps) to list unused dependencies.

- [`expand`](https://github.com/dtolnay/cargo-expand) to dump what macros expand into.

- [`asm`](https://github.com/gnzlbg/cargo-asm) to dump assembly or LLVM IR (the latter via `cargo llvm-ir`).

- [`llvm-lines`](https://github.com/dtolnay/cargo-llvm-lines) to inspect code bloat.

[Tokio Console](https://tokio.rs/blog/2021-12-announcing-tokio-console) is also interesting.

## Tips and Tricks

### `cargo watch`

During development it can be useful to have `cargo watch` automatically re-launch eg reindeer on code changes. This can be achieved as follows, in this case for Rudolph:

```sh
cargo watch -c -x 'run --bin rudolph -- --identity "localhost:50000" --port 50000 --sessions ./examples' -i examples
```

Note that `-i examples` means workers are not re-launched when files in `./examples` are changed.
