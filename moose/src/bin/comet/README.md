
# Comet

Reindeer using gRPC choreography, in-memory storage, and gRPC networking.

To launch Comet you need to specify:

- The identity to use for this particular instance; this must be interpretable as a valid gRPC endpoint.

- The port on which to start a gRPC server.

In order to run Comet with gRPC over TLS, first generate and distribute certificates to each instance, and then specify their location using the `--certs` argument. You must also specify the identity used by the choreographer.

Due to security, Comet will refuse to run with the same session id more than once. For this reason, the `cometctl` tool allows you to specify a session id using the `--session-id` parameter.

## Example

The following launches three instances:

```sh
comet \
  --identity localhost:50000 \
  --port 50000

comet \
  --identity localhost:50001 \
  --port 50001

comet \
  --identity localhost:50002 \
  --port 50002
```

and then uses the CLI tools to launch one of the sessions in the `examples` directory:

```sh
cometctl run examples/test.session
cometctl run --session-id "second session" examples/test.session
```

To run the example over TLS, using the _insecure_ certificates provided in `examples/certs`:

```sh
comet \
  --identity localhost:50000 \
  --port 50000 \
  --certs examples/certs \
  --choreographer choreographer
```

and

```sh
cometctl \
  --identity choreographer \
  --certs examples/certs \
  run examples/test.session
```
