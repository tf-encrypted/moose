alpine-comet
====
Minimal docker image for running a `comet` worker

## Usage

From repo root:

```sh
docker build -f examples/alpine-comet/Dockerfile.build . -t build-comet
cd examples
docker run -v alpine-comet:/build --rm -it build-comet cp /usr/local/cargo/bin/comet /build
```

You should see a new `comet` binary in the `examples/alpine/comet` directory. The binary is compiled for target `x86-