# Moose
Moose is a framework for secure multi-party computation, which offers an easy to use numpy-like Python API that allows users to build machine learning models that perform inference on encrypted data.


### Threat model at Cape: honest, but curious
It is permissible that if parties collude with each other, they can use result values to reconstruct inputs
This threat model is less strict than active security, which takes into account that perhaps not every party follows given protocols

### Moose has 3 (main) components: 
- EDSL (embedded domain specific language) - PyMoose
- Elk compiler
- Moose runtime

### Running onnx with PyMoose
Given an onnx file as input, a Moose computation is produced as follows:
- Convert onnx to PyMoose predictor
- PyMoose predictor builds an edsl computation using its `predictor_factory` method
- edsl computation is traced by ASTTracer into a python Computation object
- Python Computation serialized to msgpack; msgpack deserialized into Moose’s Computation struct
- Dump Moose Computation to textual format; output the result
  
### Interface between PyMoose and Moose
- pymoose/pymoose/computation is the interface between PyMoose and Moose
- pymoose/pymoose/computation/utils.py specifies how to serialize PyMoose computations (Python) to msgpack
- moose/src/computations.rs deserializes msgpack to Moose computations, matching Python operators, values, and types with Rust analogues
A computation is represented as a graph, where nodes = operators and edges = Values. Each edge has a Rust type and a Moose type  

### Levels of computations/Lowering
- Lowering is implemented via multiple dispatch of “kernel” functions
- Logical level (Python edsl) = top level
- Lower from top level through intermediate representations (IR) levels to reach the bottom level (like traversing a tree in depth first search manner)
- The final level is the runtime kernel, which is a (usually pure) Rust function/closure that can be executed at runtime
- Each “level” is represented by a particular kernel; kernels are loosely organized into “dialects” (similar to Tensorflow MLIR)

    A level can, for example, relate to placement.
    Example: The logical level is higher than the “replicated” level, which defines replicated protocol implementations
    Example: The replicated level is higher than the “host” level, because replicated protocols consist of collections of host operations

    A level can also relate to different abstractions of values/types
    Example: Floatingpoint is an abstraction over Host/Mirrored implementations of floatingpoint ops.
    Example: HostFixedTensor is higher than HostRingTensor, because it consists of a HostRingTensor with additional metadata (integral/fractional precision)

### Placements
- Host = data is on just 1 machine
- Mirrored = data is on several machines (the machines have the same data and perform the same operations to represent public data)
- Replicated = data is secret shared across several machines
- Additive = alternative secret sharing scheme, currently only used in subprotocols for replicated placement

### Truncation
Numbers are represented using fixed point representation: 128 bits for integer and fractional part.
Results of operations on fixed point numbers must be truncated in some cases  (e.g.: multiplication) to ensure that the result has the same number integer and fractional bits as the inputs. I.e.: fixed point representation must match for inputs and outputs. 

### Sessions
There are 3 types of Sessions based on execution patterns:
- Asynchronous (“prod”) 
    Performs lowering until a runtime kernel is reached, then schedule a Tokio task
- Synchronous (“dev”)
- Symbolic (“graph analysis and optimization”) 
    One can leverage symbolic execution to lower a computation without actually executing it (i.e.: run the computation with symbolic values instead of concrete values for its args). Depth-first lowering stops when a runtime kernel is reached; return computation graph after full graph has been lowered.

### SymbolicStrategy
More flexible API for symbolic execution. Allows the user to drop out of the usual lowering process used by the Sync/Async Session structs, e.g. to enforce graph-wide properties or make edits to the graph before continuing lowering
API is currently incomplete/undefined, but will be designed to be convenient to write compile-time optimizations in the future

### Runtime session attributes
Role = host placement
Identity = the host assigned to the role (e.g.: a string (Alice), IP address with a port)
