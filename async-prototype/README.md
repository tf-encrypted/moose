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

To run the example `main_with_grcp.py` over grcp, you have to start in differnt terminals 3 servers:
```
python launch_servers.py --port 50051
python launch_servers.py --port 50052
python launch_servers.py --port 50053
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
