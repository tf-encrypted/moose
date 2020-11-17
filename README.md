## Development

You will need a working [installation of Rust](https://www.rust-lang.org/learn/get-started) to compile and test this project.

We compile and test against the nightly toolchain so make sure to either set the nightly toolchain as the default using `rustup default nightly` (recommended) or run every command with `+nightly`.

We require code to be formatted according to `cargo fmt` so make sure to run this command before submitted your work. You should also run `cargo clippy` to lint your code.

To ease your development we encourage you to install the following extra cargo commands:

- [`cargo watch`](https://crates.io/crates/cargo-watchcargo-watch) will type check your code on every save;  `cargo watch --exec test` will run all tests on every save.

- [`cargo outdated`](https://crates.io/crates/cargo-outdated) checks if your dependencies are up to date.

- [`cargo audit`](https://crates.io/crates/cargo-audit) checks if any vulnerabilities have been detected for your current dependencies.

- [`cargo release`](https://crates.io/crates/cargo-release) automates the release cycle, including bumping versions.
