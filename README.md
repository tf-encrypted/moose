# Prototype Runtime

## Installation

### Bootstrapping

Install dependencies:

```
make pydep
```

Install Python library:
```
make pylib
```

These two can be wrapped into one command, if desired:
```
make install
```

Compile protobuf files:

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
