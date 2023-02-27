# Tutorials Preamble
Tutorials relating to Moose and Pymoose.

## Use Case Tutorials
These tutorials illustrate how you can use `PyMoose` to solve various use cases such as encrypted scientific computation across multiple data owners to encrypted machine learning inference.

### Running
Each tutorial has a section running the example with the `pm.LocalMooseRuntime` runtime which locally simulates this computation running across hosts. And another section running the same example over the network with the `pm.GrpcMooseRuntime` runtime. If you run the example with gRPC make sure to launch three workers with the right endpoints as follow:
```
cargo run --bin comet -- --identity localhost:50000 --port 50000
cargo run --bin comet -- --identity localhost:50001 --port 50001
cargo run --bin comet -- --identity localhost:50002 --port 50002
```

### Dependencies
To run some of these notebooks, you may have to install the additional following dependencies to your virtual environment:
```
onnxmltools==1.11.0
scikit-learn==1.0.2
skl2onnx==1.11.2
```

## Advanced Moose Usage
Some features of Moose do not have direct Python bindings in PyMoose. The rest of these tutorials show how one can use these extra Moose features while still using PyMoose to describe and generate their computations.
