# Prototype Runtime

## Installation

### Bootstrapping

```
pip install -r requirements-dev.txt
```

### Running

```
make run
```

To run the example `main_with_grcp.py` over grcp you need to first generate the grpc python files:
```
python -m grpc_tools.protoc \
      --proto_path=. \
      --python_out=. \
      --grpc_python_out=. \
      protos/*.proto
```

Then start the workers using Docker:
```
cd docker/dev
docker-compose up
```

Then run:
``` 
python main_with_grpc.py
```

### Developing

We do not have any CI set up for this prototype so be careful to run all of the following commands before committing:

```
make fmt
make lint
make test
```
