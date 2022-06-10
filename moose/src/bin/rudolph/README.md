
# Rudolph

Reindeer using filesystem choreography, in-memory storage, and gRPC networking.

To launch Rudolph you need to specify:

- The identity to use for this particular instance; this must be a interpretable as a valid gRPC endpoint.

- The port on which to start a gRPC server; this must match the given identity.

- The directory in which to look for sessions.

The following is an example using the provided `examples` directory for sessions:

```sh
rudolph \
  --identity "localhost:50000" \
  --port 50000 \
  --sessions ./examples
```

During development it can be useful to have `cargo watch` automatically re-launch workers on code changes. This can be achieved as follows.

```sh
cargo watch -c -x 'run --bin rudolph -- --identity "localhost:50000" --port 50000 --sessions ./examples' -i examples
```

Note that `-i examples` means workers are not re-launched when files in `./examples` are changed.

## Logging

Debug logging to console may be turned on by setting:

```sh
export RUST_LOG="moose=debug"
```

## Tracing

Rudolph has basic support for Jaeger/OpenTelemetry via the `--telemetry` flag.

Jaeger may be launched via docker:

```sh
docker run \
  -p6831:6831/udp \
  -p6832:6832/udp \
  -p16686:16686 \
  jaegertracing/all-in-one:latest
```

We encourage setting `RUST_LOG` to an appropriate value to limit the number of spans generated, in particular for large computations.

## Heaptrack

We have been using [heaptrack](https://github.com/KDE/heaptrack) to measure the memory consumptions of computations,
but in order to do so it appears that you need to launch Rudolph directly, and not via `cargo run`.

This can be done by first building it:

```sh
cargo build --bin rudolph
```

and then running the produced executable:

```sh
./target/debug/rudolph
```

In particular, to use heaptrack run:

```sh
heaptrack ./target/debug/rudolph
```

# TLS support

In order to run Rudolph with gRPC over TLS make sure to generate TLS certificates.

The (insecure) certificates used for `test.session` example were generated as follows using [certstrap](https://github.com/square/certstrap):

```sh
certstrap --depot-path certs init --common-name ca --passphrase ""

certstrap --depot-path certs request-cert --common-name choreographer --domain localhost --passphrase ""
certstrap --depot-path certs sign --CA ca choreographer

certstrap --depot-path certs request-cert --common-name localhost:50000 --domain localhost --passphrase ""
certstrap --depot-path certs sign --CA ca localhost_50000

certstrap --depot-path certs request-cert --common-name localhost:50001 --domain localhost --passphrase ""
certstrap --depot-path certs sign --CA ca localhost_50001

certstrap --depot-path certs request-cert --common-name localhost:50002 --domain localhost --passphrase ""
certstrap --depot-path certs sign --CA ca localhost_50002
```

To run `test.moose` using `rudolph`, type the following commands in individual terminals inside the:

```sh
rudolph \
  --identity localhost:50000 \
  --port 50000 \
  --session ./examples \
  --certs examples/certs

rudolph \
  --identity localhost:50001 \
  --port 50001 \
  --session ./examples \
  --certs examples/certs

rudolph \
  --identity localhost:50002 \
  --port 50002 \
  --session ./examples \
  --certs examples/certs
```
