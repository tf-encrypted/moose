# Reindeer

Collection of workers intended for testing and benchmarking purposes. They also illustrate our LEGO approach of putting together custom workers by mixing networking, storage, and choreography [modules](../modules).

**Warning: This code is not intended for production settings.**

# Dasher

TODO

# Rudolph

Worker using filesystem-based choreography, in-memory storage, and gRPC-based networking.

## Launching

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

## Sessions

Sessions are controlled by two sets of files:

- `.moose` files that contain Moose computations in either textual or binary format. These must be physical computations since no compilation is currently done by the worker.

- `.session` files that specify sessions and we encourage taking a look at the example files to see their format. The name of the file is used to derive the session id.

The worker listens for changes to the specified directory and will automatically launch new sessions when new `.session` files are created, i.e. there is no need to relaunch workers to run new sessions.

## Logging

Debug logging to console may be turned on by setting:

```
export RUST_LOG="moose=debug,moose_modules=debug"
```

## Tracing

Rudolph has basic support for Jaeger/OpenTelemetry via the `--telemetry` flag.

Jaeger may be launched via docker:

```
docker run -p6831:6831/udp -p6832:6832/udp -p16686:16686 jaegertracing/all-in-one:latest
```

Once running, Rudolph can be launched via the following. Note that we encourage setting `RUST_LOG` to limit the number of spans generated, in particular for large comptuations.

```
cargo run --bin rudolph -- --identity "localhost:50000" --port 50000 --sessions ./examples --telemetry
```

## Heaptrack

We have been using [heaptrack](https://github.com/KDE/heaptrack) to measure the memory consumptions of computations,
but in order to do so it appears that you need to launch Rudolph directly, and not via `cargo run`.

This can be done by first building it:

```
cargo build --bin rudolph
```

and then running the produced executable:

```
./target/debug/rudolph
```

In particular, to use heaptrack run:

```
heaptrack ./target/debug/rudolph
```

# TLS support

In order to run the reindeer with gRPC and TLS make sure to generate TLS certificates.
The certificates used for `test.session` example were generated using the following commands.

```
cd examples
certstrap --depot-path certs init --common-name ca --passphrase ""
certstrap --depot-path certs request-cert --common-name choreographer --domain localhost --passphrase ""
certstrap --depot-path certs request-cert --common-name localhost:50000 --domain localhost --passphrase ""
certstrap --depot-path certs request-cert --common-name localhost:50001 --domain localhost --passphrase ""
certstrap --depot-path certs request-cert --common-name localhost:50002 --domain localhost --passphrase ""
certstrap --depot-path certs sign --CA ca choreographer
certstrap --depot-path certs sign --CA ca localhost_50000
certstrap --depot-path certs sign --CA ca localhost_50001
certstrap --depot-path certs sign --CA ca localhost_50002
```

To run `test.moose` using `rudolph`, type the following commands in individual terminals inside the `reindeer` folder:

```
cargo run --bin rudolph -- --identity 'localhost:50000' --port 50000 --session ./examples --no-listen --certs examples/certs
cargo run --bin rudolph -- --identity 'localhost:50001' --port 50001 --session ./examples --no-listen --certs examples/certs
cargo run --bin rudolph -- --identity 'localhost:50002' --port 50002 --session ./examples --no-listen --certs examples/certs
```

To run `test.moose` using `comet`, type the following commands in individual terminals inside the `reindeer` folder:

```
cargo run --bin comet -- --identity localhost:50000 --port 50000 --certs examples/certs --choreographer choreographer
cargo run --bin comet -- --identity localhost:50001 --port 50001 --certs examples/certs --choreographer choreographer
cargo run --bin comet -- --identity localhost:50002 --port 50002 --certs examples/certs --choreographer choreographer
cargo run --bin cometctl -- --identity choreographer --certs examples/certs run examples/test.session
```
