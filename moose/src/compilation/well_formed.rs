use crate::computation::{Computation, Operator};
use crate::execution::SymbolicSession;
use crate::kernels::{DispatchKernel, NgDispatchKernel};
use crate::Error;
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
        let compile_error: Option<Error> = match &op.kind {
            // TODO(Morten) use DispatchKernel::compile for these as well
            Load(_) | Save(_) | Send(_) | Receive(_) => None,

            Shape(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Broadcast(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            PrfKeyGen(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Xor(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            And(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Or(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            BitExtract(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Shl(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            ShlDim(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Shr(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Sample(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            SampleSeeded(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            RingFixedpointAbs(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            RingFixedpointArgmax(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            RingFixedpointMean(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            RingFixedpointEncode(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            RingFixedpointDecode(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            RingInject(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Fill(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Share(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Reveal(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            TruncPr(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Msb(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            RepToAdt(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            BitDecompose(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            BitCompose(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            AdtToRep(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            DeriveSeed(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Constant(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Input(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Output(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            AtLeast2D(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            FixedpointEncode(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            FixedpointDecode(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Sign(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Transpose(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Squeeze(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Identity(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Cast(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Reshape(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Slice(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Ones(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            ExpandDims(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Concat(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Dot(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Inverse(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Add(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Sub(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Mul(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Mean(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Sum(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Div(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            AddN(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Exp(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Pow2(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Neg(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Log(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Log2(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Equal(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            EqualZero(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Mux(op) => NgDispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            LessThan(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            GreaterThan(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            IndexAxis(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Index(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Sigmoid(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Maximum(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Softmax(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Argmax(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Demirror(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Mirror(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Decrypt(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Sqrt(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Abs(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
            Diag(op) => DispatchKernel::<SymbolicSession>::compile(op, plc).err(),
        };
        if let Some(e) = compile_error {
            return Err(e.into());
        }
    }

    Ok(comp)
}
