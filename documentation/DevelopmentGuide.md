# Moose
Moose is a framework for secure multi-party computation, which offers an easy to use numpy-like Python API that allows users to build machine learning models that perform inference on encrypted data.


### How to implement new protocols in Moose
Example: How to implement an operation edsl.ones(), which creates a vector of ones with a given lenght

Summary
- pymoose/pymoose/edsl/base.py

    Add class OnesExperssion

    Add def ones(), which returns OnesExpression(placement, inputs, vtype)

- pymoose/pymoose/edsl/tracer.py

    Add def visit_OnesExpression to class AstTracer, which returns a logical computation (by mapping expressions to operations that are added to computational graph

- pymoose/pymoose/computation/utils.py

    Add ops.OnesOperation to SUPPORTED_TYPES

    This script takes a computational graph and serializes it using msgpack to output a binary that can be passed into Rust

- pymoose/pymoose/computation/operations.py

    Add class OnesOperation

- pymoose/src/bindings.py

    Nothing to add

    These are the bindings between Python and Rust

- pymoose/src/computation.rs

    Add OnesOperation(PyOnesOperation) to enum PyOperation

    Add struct PyOnesOperation(name, inputs, placement_name, signature)

    Add OnesOperation to TryFrom

- moose/src/computation.rs

    Add Ones to operators![]

    Add pub struct OnesOp

- moose/src/compilation/well-formed.rs

    Add Ones(op) => DispatchKernel

- moose/src/execution/asynchronous.rs, symbolic.rs, synchronous.rs

    Add Ones(op) => DispatchKernel

- moose/src/floatingpoint/ops.rs

    Add impl OnesOp

- moose/src/host/ops.rs (if the protocol is for host placements)

    Add impl OnesOp
