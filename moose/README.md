# Moose

Moose is the core framework for encrypted learning and data processing in distributed environments. It is production ready and secure by default.

## Running

Besides a recent [rust stable](https://rust-lang.github.io/rustup/concepts/channels.html) you only nee

Install OpenBLAS development headers via `libopenblas-dev` for Ubuntu.

### Developing

### Testing

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

Moose is distributed under the terms of Apache License (Version 2.0). Copyright as specified in [NOTICE](../NOTICE).
