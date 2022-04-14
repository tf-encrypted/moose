use crate::computation::{Computation, Operator};
use crate::execution::SymbolicSession;
use crate::kernels::{DispatchKernel, Kernel};
use std::collections::HashSet;

/// Perform basic well-formed check of computation without modification.
///
/// Note that this check is not completely sound wrt to runtime errors:
/// - some unsupported operator instantiations are currently only checked at runtime
/// - some potential ndarray errors cannot currently be caught statically
/// - computations may only be partially specified, for instance around shapes
pub fn well_formed(comp: Computation) -> anyhow::Result<Computation> {
    let mut seen_values: HashSet<&String> = HashSet::with_capacity(comp.operations.len());

    for op in &comp.operations {
        // Make sure computation is in topological order
        for input_op_name in &op.inputs {
            if !seen_values.contains(input_op_name) {
                return Err(crate::Error::MalformedEnvironment(input_op_name.to_string()).into());
            }
        }
        seen_values.insert(&op.name);

        // Make sure computation only contains valid operator instantiations
        // by attempting to compile (symbolic) kernels for them
        use Operator::*;
        let plc = &op.placement;
        let _kernel: Kernel<SymbolicSession> = match &op.kind {
            // TODO(Morten) use DispatchKernel::compile for these as well
            Load(_) | Save(_) | Send(_) | Receive(_) => {
                let kernel: Kernel<_> = Box::new(|_, _| unimplemented!());
                Ok(kernel)
            }

            Shape(op) => DispatchKernel::compile(op, plc),
            Broadcast(op) => DispatchKernel::compile(op, plc),
            PrfKeyGen(op) => DispatchKernel::compile(op, plc),
            Xor(op) => DispatchKernel::compile(op, plc),
            And(op) => DispatchKernel::compile(op, plc),
            Or(op) => DispatchKernel::compile(op, plc),
            BitExtract(op) => DispatchKernel::compile(op, plc),
            Shl(op) => DispatchKernel::compile(op, plc),
            ShlDim(op) => DispatchKernel::compile(op, plc),
            Shr(op) => DispatchKernel::compile(op, plc),
            Sample(op) => DispatchKernel::compile(op, plc),
            SampleSeeded(op) => DispatchKernel::compile(op, plc),
            RingFixedpointAbs(op) => DispatchKernel::compile(op, plc),
            RingFixedpointArgmax(op) => DispatchKernel::compile(op, plc),
            RingFixedpointMean(op) => DispatchKernel::compile(op, plc),
            RingFixedpointEncode(op) => DispatchKernel::compile(op, plc),
            RingFixedpointDecode(op) => DispatchKernel::compile(op, plc),
            RingInject(op) => DispatchKernel::compile(op, plc),
            Fill(op) => DispatchKernel::compile(op, plc),
            Share(op) => DispatchKernel::compile(op, plc),
            Reveal(op) => DispatchKernel::compile(op, plc),
            TruncPr(op) => DispatchKernel::compile(op, plc),
            Msb(op) => DispatchKernel::compile(op, plc),
            RepToAdt(op) => DispatchKernel::compile(op, plc),
            BitDecompose(op) => DispatchKernel::compile(op, plc),
            BitCompose(op) => DispatchKernel::compile(op, plc),
            AdtToRep(op) => DispatchKernel::compile(op, plc),
            DeriveSeed(op) => DispatchKernel::compile(op, plc),
            Constant(op) => DispatchKernel::compile(op, plc),
            Input(op) => DispatchKernel::compile(op, plc),
            Output(op) => DispatchKernel::compile(op, plc),
            AtLeast2D(op) => DispatchKernel::compile(op, plc),
            FixedpointEncode(op) => DispatchKernel::compile(op, plc),
            FixedpointDecode(op) => DispatchKernel::compile(op, plc),
            Sign(op) => DispatchKernel::compile(op, plc),
            Transpose(op) => DispatchKernel::compile(op, plc),
            Squeeze(op) => DispatchKernel::compile(op, plc),
            Identity(op) => DispatchKernel::compile(op, plc),
            Cast(op) => DispatchKernel::compile(op, plc),
            Reshape(op) => DispatchKernel::compile(op, plc),
            Slice(op) => DispatchKernel::compile(op, plc),
            Ones(op) => DispatchKernel::compile(op, plc),
            Zeros(op) => DispatchKernel::compile(op, plc),
            ExpandDims(op) => DispatchKernel::compile(op, plc),
            Concat(op) => DispatchKernel::compile(op, plc),
            Dot(op) => DispatchKernel::compile(op, plc),
            Inverse(op) => DispatchKernel::compile(op, plc),
            Add(op) => DispatchKernel::compile(op, plc),
            Sub(op) => DispatchKernel::compile(op, plc),
            Mul(op) => DispatchKernel::compile(op, plc),
            Mean(op) => DispatchKernel::compile(op, plc),
            Sum(op) => DispatchKernel::compile(op, plc),
            Div(op) => DispatchKernel::compile(op, plc),
            AddN(op) => DispatchKernel::compile(op, plc),
            ReLU(op) => DispatchKernel::compile(op, plc),
            Exp(op) => DispatchKernel::compile(op, plc),
            Pow2(op) => DispatchKernel::compile(op, plc),
            Neg(op) => DispatchKernel::compile(op, plc),
            Log(op) => DispatchKernel::compile(op, plc),
            Log2(op) => DispatchKernel::compile(op, plc),
            Equal(op) => DispatchKernel::compile(op, plc),
            EqualZero(op) => DispatchKernel::compile(op, plc),
            Mux(op) => DispatchKernel::compile(op, plc),
            LessThan(op) => DispatchKernel::compile(op, plc),
            GreaterThan(op) => DispatchKernel::compile(op, plc),
            IndexAxis(op) => DispatchKernel::compile(op, plc),
            Index(op) => DispatchKernel::compile(op, plc),
            Sigmoid(op) => DispatchKernel::compile(op, plc),
            Maximum(op) => DispatchKernel::compile(op, plc),
            Softmax(op) => DispatchKernel::compile(op, plc),
            Argmax(op) => DispatchKernel::compile(op, plc),
            Demirror(op) => DispatchKernel::compile(op, plc),
            Mirror(op) => DispatchKernel::compile(op, plc),
            Decrypt(op) => DispatchKernel::compile(op, plc),
            Sqrt(op) => DispatchKernel::compile(op, plc),
            Abs(op) => DispatchKernel::compile(op, plc),
            Diag(op) => DispatchKernel::compile(op, plc),
        }?;
    }

    Ok(comp)
}
