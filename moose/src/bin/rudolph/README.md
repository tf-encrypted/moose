# Rudolph

Reindeer using filesystem choreography, in-memory storage, and gRPC networking.

To launch Rudolph you need to specify:

- The identity to use for this particular instance; this must be interpretable as a valid gRPC endpoint.

- The port on which to start a gRPC server; this must match the given identity.

- The directory in which to look for sessions.

In order to run Rudolph with gRPC over TLS, first generate and distribute certificates to each instance, and then specify their location using the `--certs` argument.

## Example

The following launches three instances using the session files in the `examples` directory:

```sh
rudolph \
  --identity localhost:50000 \
  --port 50000 \
  --sessions ./examples

rudolph \
  --identity localhost:50001 \
  --port 50001 \
  --sessions ./examples

rudolph \
  --identity localhost:50002 \
  --port 50002 \
  --sessions ./examples
```

To launch a new session, simply distribute new `.session` (and associated `.moose`) files:

```sh
cp ./examples/test.session ./examples/test2.session
```

To run the example over TLS, using the _insecure_ certificates provided in `examples/certs`:

```sh
rudolph \
  --identity localhost:50000 \
  --port 50000 \
  --sessions ./examples \
  --certs examples/certs
```
