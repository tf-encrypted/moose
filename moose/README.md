# Moose

Moose is a production-ready Rust framework for encrypted learning and data processing in a distributed environment.

TODO: improve example below, and add as proper example; include in cargo docs?

```rust
use moose::prelude::*;

let alice = HostPlacement::from("alice");
let rep = ReplicatedPlacement::from(["alice", "bob", "carole"]);

let sess = SyncSession::default();

let x: HostRingTensor<_> = alice.from_raw(x);
let x_shared = rep.share(&sess, &x);
let argmax_shared = rep.argmax(&sess, axis, upmost_index, &x_shared);

let argmax = alice.reveal(&sess, &argmax_shared);
let y_target: HostRing64Tensor = alice.from_raw(y_target);
```

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

## License

Moose is distributed under the terms of Apache License (Version 2.0). Copyright as specified in [NOTICE](../NOTICE).
