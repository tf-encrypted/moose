# Prototype Runtime

## Installation

### Bootstrapping

Install dependencies:

```
pip install -r requirements-dev.txt
```

Install library:
```
pip install -e .
```

Compile protobuf files:

```
python -m grpc_tools.protoc \
      --proto_path=. \
      --python_out=. \
      --grpc_python_out=. \
      protos/*.proto
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

We do not have any CI set up for this prototype so be careful to run all of the following commands before committing:

```
make fmt
make lint
make test
```
