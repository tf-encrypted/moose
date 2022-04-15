use crate::computation::{Computation, Operator, SymbolicValue};
use crate::execution::SymbolicSession;
use crate::kernels::NgDispatchKernel;
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

            Shape(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Broadcast(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            PrfKeyGen(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            Xor(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            And(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Or(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            BitExtract(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            Shl(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            ShlDim(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            Shr(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Sample(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            SampleSeeded(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            RingFixedpointAbs(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            RingFixedpointArgmax(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            RingFixedpointMean(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            RingFixedpointEncode(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            RingFixedpointDecode(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            RingInject(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            Fill(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Share(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Reveal(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            TruncPr(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            Msb(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            RepToAdt(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            BitDecompose(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            BitCompose(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            AdtToRep(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            DeriveSeed(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            Constant(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            Input(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Output(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            AtLeast2D(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            FixedpointEncode(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            FixedpointDecode(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            Sign(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Transpose(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            Squeeze(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            Identity(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            Cast(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Reshape(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            Slice(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Ones(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            ExpandDims(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            Concat(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            Dot(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Inverse(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            Add(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Sub(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Mul(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Mean(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Sum(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Div(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            AddN(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Exp(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Pow2(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Neg(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Log(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Log2(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Equal(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            EqualZero(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            Mux(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            LessThan(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            GreaterThan(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            IndexAxis(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            Index(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Sigmoid(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            Maximum(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            Softmax(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            Argmax(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            Demirror(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            Mirror(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            Decrypt(op) => {
                NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err()
            }
            Sqrt(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Abs(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Diag(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
            Zeros(op) => NgDispatchKernel::<SymbolicSession, SymbolicValue>::compile(op, plc).err(),
        };
        if let Some(e) = compile_error {
            return Err(e.into());
        }
    }

    Ok(comp)
}
