# Moose

Core Moose framework and binaries.

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

## Binaries

