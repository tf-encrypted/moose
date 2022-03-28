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
        let op_name = op.name.as_str();
        let _kernel: Kernel<SymbolicSession> = match &op.kind {
            // TODO(Morten) use DispatchKernel::compile for these as well
            Load(_) | Save(_) | Send(_) | Receive(_) => {
                let kernel: Kernel<_> = Box::new(|_, _| unimplemented!());
                Ok(kernel)
            }

            Shape(op) => DispatchKernel::compile(op, plc, op_name),
            Broadcast(op) => DispatchKernel::compile(op, plc, op_name),
            PrfKeyGen(op) => DispatchKernel::compile(op, plc, op_name),
            Xor(op) => DispatchKernel::compile(op, plc, op_name),
            And(op) => DispatchKernel::compile(op, plc, op_name),
            Or(op) => DispatchKernel::compile(op, plc, op_name),
            BitExtract(op) => DispatchKernel::compile(op, plc, op_name),
            Shl(op) => DispatchKernel::compile(op, plc, op_name),
            ShlDim(op) => DispatchKernel::compile(op, plc, op_name),
            Shr(op) => DispatchKernel::compile(op, plc, op_name),
            Sample(op) => DispatchKernel::compile(op, plc, op_name),
            SampleSeeded(op) => DispatchKernel::compile(op, plc, op_name),
            RingFixedpointAbs(op) => DispatchKernel::compile(op, plc, op_name),
            RingFixedpointArgmax(op) => DispatchKernel::compile(op, plc, op_name),
            RingFixedpointMean(op) => DispatchKernel::compile(op, plc, op_name),
            RingFixedpointEncode(op) => DispatchKernel::compile(op, plc, op_name),
            RingFixedpointDecode(op) => DispatchKernel::compile(op, plc, op_name),
            RingInject(op) => DispatchKernel::compile(op, plc, op_name),
            Fill(op) => DispatchKernel::compile(op, plc, op_name),
            Share(op) => DispatchKernel::compile(op, plc, op_name),
            Reveal(op) => DispatchKernel::compile(op, plc, op_name),
            TruncPr(op) => DispatchKernel::compile(op, plc, op_name),
            Msb(op) => DispatchKernel::compile(op, plc, op_name),
            RepToAdt(op) => DispatchKernel::compile(op, plc, op_name),
            BitDecompose(op) => DispatchKernel::compile(op, plc, op_name),
            BitCompose(op) => DispatchKernel::compile(op, plc, op_name),
            AdtToRep(op) => DispatchKernel::compile(op, plc, op_name),
            DeriveSeed(op) => DispatchKernel::compile(op, plc, op_name),
            Constant(op) => DispatchKernel::compile(op, plc, op_name),
            Input(op) => DispatchKernel::compile(op, plc, op_name),
            Output(op) => DispatchKernel::compile(op, plc, op_name),
            AtLeast2D(op) => DispatchKernel::compile(op, plc, op_name),
            FixedpointEncode(op) => DispatchKernel::compile(op, plc, op_name),
            FixedpointDecode(op) => DispatchKernel::compile(op, plc, op_name),
            Sign(op) => DispatchKernel::compile(op, plc, op_name),
            Transpose(op) => DispatchKernel::compile(op, plc, op_name),
            Squeeze(op) => DispatchKernel::compile(op, plc, op_name),
            Identity(op) => DispatchKernel::compile(op, plc, op_name),
            Cast(op) => DispatchKernel::compile(op, plc, op_name),
            Reshape(op) => DispatchKernel::compile(op, plc, op_name),
            Slice(op) => DispatchKernel::compile(op, plc, op_name),
            Ones(op) => DispatchKernel::compile(op, plc, op_name),
            ExpandDims(op) => DispatchKernel::compile(op, plc, op_name),
            Concat(op) => DispatchKernel::compile(op, plc, op_name),
            Dot(op) => DispatchKernel::compile(op, plc, op_name),
            Inverse(op) => DispatchKernel::compile(op, plc, op_name),
            Add(op) => DispatchKernel::compile(op, plc, op_name),
            Sub(op) => DispatchKernel::compile(op, plc, op_name),
            Mul(op) => DispatchKernel::compile(op, plc, op_name),
            Mean(op) => DispatchKernel::compile(op, plc, op_name),
            Sum(op) => DispatchKernel::compile(op, plc, op_name),
            Div(op) => DispatchKernel::compile(op, plc, op_name),
            AddN(op) => DispatchKernel::compile(op, plc, op_name),
            Exp(op) => DispatchKernel::compile(op, plc, op_name),
            Pow2(op) => DispatchKernel::compile(op, plc, op_name),
            Neg(op) => DispatchKernel::compile(op, plc, op_name),
            Log(op) => DispatchKernel::compile(op, plc, op_name),
            Log2(op) => DispatchKernel::compile(op, plc, op_name),
            Equal(op) => DispatchKernel::compile(op, plc, op_name),
            EqualZero(op) => DispatchKernel::compile(op, plc, op_name),
            Mux(op) => DispatchKernel::compile(op, plc, op_name),
            LessThan(op) => DispatchKernel::compile(op, plc, op_name),
            GreaterThan(op) => DispatchKernel::compile(op, plc, op_name),
            IndexAxis(op) => DispatchKernel::compile(op, plc, op_name),
            Index(op) => DispatchKernel::compile(op, plc, op_name),
            Sigmoid(op) => DispatchKernel::compile(op, plc, op_name),
            Maximum(op) => DispatchKernel::compile(op, plc, op_name),
            Softmax(op) => DispatchKernel::compile(op, plc, op_name),
            Argmax(op) => DispatchKernel::compile(op, plc, op_name),
            Demirror(op) => DispatchKernel::compile(op, plc, op_name),
            Mirror(op) => DispatchKernel::compile(op, plc, op_name),
            Decrypt(op) => DispatchKernel::compile(op, plc, op_name),
            Sqrt(op) => DispatchKernel::compile(op, plc, op_name),
            Abs(op) => DispatchKernel::compile(op, plc, op_name),
            Diag(op) => DispatchKernel::compile(op, plc, op_name),
        }?;
    }

    Ok(comp)
}
