# Reindeer

Collection of workers intended for testing and benchmarking purposes. They also illustrate our LEGO approach of putting together custom workers by mixing networking, storage, and choreography [modules](../modules).

**Warning: This code is not intended for production settings.**

# Dasher

TODO

# Rudolph

Worker using filesystem-based choreography, in-memory storage, and gRPC-based networking.

## Launching workers

To launch Rudolph you need to specify:

- The identity to use for this particular worker; this string must be a interpretable as a valid gRPC endpoint with an implicit `http://` prefix.

- The port on which to start a gRPC server.

- The directory to look for sessions in (see below).

The following is an example using the provided `examples` directory for sessions:

```
cargo run --bin rudolph -- --identity "localhost:50000" --port 50000 --sessions ./examples
```

Note that during development it can be useful to have `cargo watch` automatically re-launch workers on code changes. This can be achieved as follows.

```
cargo watch -c -x 'run --bin rudolph -- --identity "localhost:50000" --port 50000 --sessions ./examples' -i examples
```

Note that the `-i examples` makes sure that the workers are not re-launched when files in `./examples` are changed.

Debug logging may be turned on by setting:

```
export RUST_LOG="worker=debug"
```

## Launching sessions

Sessions are controlled by two sets of files:

- `.moose` files that contain Moose computations in either textual or binary format. These must be physical computations since no compilation is currently done by the worker.

- `.session` files that specify sessions and we encourage taking a look at the example files to see their format. The name of the file is used to derive the session id.

The worker listens for changes to the specified directory and will automatically launch new sessions when new `.session` files are created, i.e. there is no need to relaunch workers to run new sessions.
