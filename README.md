# Prototype Runtime

## Installation

### Bootstrapping

Install python development headers for your OS. (eg - `python3-dev` for Ubuntu, or `python38-devel` for OpenSUSE)

To install the library and all of its dependencies, run:
```
make install
```

This unwraps into two other targets, which are kept separate for purposes of caching in CI:

```
make pydep  # install dependencies
make pylib  # install runtime Python library
```

You will also need to compile protobuf files before running any examples that use gRPC, which you can do via:

```
make build
```

### Running locally for testing

```
python main.py --runtime test
```

### Running with remote cluster

You can start a cluster locally using the following:

```
cd docker/dev
docker-compose up
```

Once done you can run the following to evaluate a computation on it:

```
python main.py --runtime remote
```

### Developing

To ensure your changes will pass our CI, it's wise to run the following commands before committing:

```
make ci

# or, more verbosely:

make fmt
make lint
make test
```
