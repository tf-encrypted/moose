# Prototype Runtime

## Installation

### Bootstrapping

```
pip install -r requirements.txt
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
      "protos/secure_channel.proto"
```

Then start in differnt terminals 3 servers:
```
python launch_servers.py --port 50051 --valuename x0 --value 5
python launch_servers.py --port 50052 --valuename x1 --value 7
python launch_servers.py --port 50053
python launch_servers.py --port 50054
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
