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
