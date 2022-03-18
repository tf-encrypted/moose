//! Symbolic execution of computations
//!
//! This is used during compilation to lower operations.
//! In general, it works by evaluating kernels on symbolic values and
//! recording the underlying operations perform as new computation.
//! Values are generally wrapped in the `Symbolic` enum.

use super::{Operands, Session, SetupGeneration};
use crate::computation::*;
use crate::error::{Error, Result};
use crate::host::HostPrfKey;
use crate::kernels::{DispatchKernel, Kernel, PlacementPlace};
use crate::replicated::{RepSetup, ReplicatedPlacement};
use crate::{MirroredCounterpart, Ring, TensorLike, Underlying};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Wrapper for values used in `SymbolicSession`s
#[derive(Clone, Debug, PartialEq)]
pub enum Symbolic<T: Placed> {
    /// The value is really symbolic
    ///
    /// It exists only as a handle to an operation.
    Symbolic(SymbolicHandle<T::Placement>),

    /// The value is actually not symbolic
    ///
    /// It (partially) exists, although some sub-components may be handles
    Concrete(T),
}

impl<T: Placed> Symbolic<T> {
    #[allow(dead_code)]
    pub(crate) fn is_symbolic(&self) -> bool {
        match self {
            Symbolic::Symbolic(_) => true,
            Symbolic::Concrete(_) => false,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn is_concrete(&self) -> bool {
        match self {
            Symbolic::Symbolic(_) => false,
            Symbolic::Concrete(_) => true,
        }
    }

    pub(crate) fn symbolic_handle(&self) -> Option<&SymbolicHandle<T::Placement>> {
        match self {
            Symbolic::Symbolic(h) => Some(h),
            Symbolic::Concrete(_) => None,
        }
    }
}

impl<T: Placed> Underlying for Symbolic<T>
where
    T: Underlying,
{
    type TensorType = <T as Underlying>::TensorType;
}

impl<T: Placed> MirroredCounterpart for Symbolic<T>
where
    T: MirroredCounterpart,
    <T as MirroredCounterpart>::MirroredType: Placed,
{
    type MirroredType = Symbolic<<T as MirroredCounterpart>::MirroredType>;
}

impl<T: Placed> TensorLike for Symbolic<T>
where
    T: TensorLike,
{
    type Scalar = <T as TensorLike>::Scalar;
}

impl<T: Placed> Ring for Symbolic<T>
where
    T: Ring,
{
    type BitLength = T::BitLength;
}

#[derive(Clone, Debug, PartialEq)]
pub struct SymbolicHandle<P> {
    pub op: String,
    // NOTE if we had a handle to the graph we
    // could perhaps derive the placement instead
    pub plc: P,
}

impl<T: Placed> Placed for Symbolic<T>
where
    T::Placement: Clone,
{
    type Placement = T::Placement;

    fn placement(&self) -> Result<Self::Placement> {
        match self {
            Symbolic::Symbolic(x) => Ok(x.plc.clone()),
            Symbolic::Concrete(x) => x.placement(),
        }
    }
}

impl<S: Session, T, P> PlacementPlace<S, Symbolic<T>> for P
where
    T: Placed<Placement = P>,
    P: PlacementPlace<S, T>,
    P: Clone + PartialEq,
{
    fn place(&self, sess: &S, x: Symbolic<T>) -> Symbolic<T> {
        match x.placement() {
            Ok(ref place) if place == self => x,
            _ => {
                match x {
                    Symbolic::Concrete(x) => {
                        // TODO should we indirectly insert Place ops here?
                        let x = self.place(sess, x);
                        Symbolic::Concrete(x)
                    }
                    Symbolic::Symbolic(SymbolicHandle { op, plc: _ }) => {
                        // TODO insert `Place` ops here?
                        Symbolic::Symbolic(SymbolicHandle {
                            op,
                            plc: self.clone(),
                        })
                    }
                }
            }
        }
    }
}

#[derive(Default)]
struct SymbolicSessionState {
    pub ops: Vec<Operation>,
    pub replicated_keys: HashMap<ReplicatedPlacement, Arc<RepSetup<Symbolic<HostPrfKey>>>>,
}

/// Session object in which symbolic execution is happening
pub struct SymbolicSession {
    pub(crate) strategy: Box<dyn SymbolicStrategy>,
    state: Arc<RwLock<SymbolicSessionState>>,
}

impl Default for SymbolicSession {
    fn default() -> Self {
        SymbolicSession {
            strategy: Box::new(DefaultSymbolicStrategy),
            state: Default::default(),
        }
    }
}

impl SymbolicSession {
    /// Add operation to the session's underlying computation
    pub(crate) fn add_operation<'s, O, P, Q>(
        &'s self,
        operator: &O,
        operands: &[&str],
        plc: &P,
    ) -> SymbolicHandle<Q>
    where
        O: Into<Operator> + Clone,
        P: Into<Placement> + Clone,
        P: Into<Q>,
    {
        let mut state = self.state.write();
        let op_name: String = format!("op_{}", state.ops.len());
        let op = Operation {
            name: op_name.clone(),
            kind: operator.clone().into(),
            inputs: operands.iter().map(|op| op.to_string()).collect(),
            placement: plc.clone().into(),
        };
        state.ops.push(op);

        SymbolicHandle {
            op: op_name,
            plc: plc.clone().into(),
        }
    }

    /// Apply a given closure to the iterator over the ops.
    ///
    /// The "ops" vector is locked for READ for the duration of the call.
    #[cfg(test)]
    pub(crate) fn ops_iter<F: FnMut(std::slice::Iter<Operation>) -> T, T>(
        &self,
        mut operation: F,
    ) -> T {
        let state = self.state.read();
        operation(state.ops.iter())
    }
}

impl Session for SymbolicSession {
    type Value = crate::computation::SymbolicValue;
    fn execute(
        &self,
        op: &Operator,
        plc: &Placement,
        operands: Operands<Self::Value>,
    ) -> Result<Self::Value> {
        self.strategy.execute(self, op, plc, operands)
    }
}

impl SetupGeneration<ReplicatedPlacement> for SymbolicSession {
    type Setup = RepSetup<Symbolic<HostPrfKey>>;

    fn setup(&self, plc: &ReplicatedPlacement) -> Result<Arc<Self::Setup>> {
        // Produce a new replicated setup or returned a previously produced setup for the placement
        let state = self.state.read();
        match state.replicated_keys.get(plc) {
            Some(setup) => Ok(Arc::clone(setup)),
            None => {
                drop(state); // Release the read access

                // This may (likely) grab a write lock to the state inside
                let new_setup = plc.gen_setup(self)?;

                // Grab a new write lock.
                let mut state = self.state.write();
                // Only insert if missing, since someone else might have done that already
                // If our `new_setup` ends up being unused due to the race, it will be pruned later on by a dedicated pruning pass.
                let setup = state
                    .replicated_keys
                    .entry(plc.clone())
                    .or_insert_with(|| Arc::new(new_setup));
                Ok(Arc::clone(setup))
            }
        }
    }
}

impl DispatchKernel<SymbolicSession> for SendOp {
    fn compile(&self, _plc: &Placement) -> Result<Kernel<SymbolicSession>> {
        Err(Error::Compilation(
            "SendOp not supported on symbolic sessions".to_string(),
        ))
    }
}

impl DispatchKernel<SymbolicSession> for ReceiveOp {
    fn compile(&self, _plc: &Placement) -> Result<Kernel<SymbolicSession>> {
        Err(Error::Compilation(
            "ReceiveOp not supported on symbolic sessions".to_string(),
        ))
    }
}

pub(crate) trait SymbolicStrategy {
    fn execute(
        &self,
        sess: &SymbolicSession,
        op: &Operator,
        plc: &Placement,
        operands: Operands<SymbolicValue>,
    ) -> Result<SymbolicValue>;
}

#[derive(Clone, Copy, Debug)]
struct DefaultSymbolicStrategy;

impl SymbolicStrategy for DefaultSymbolicStrategy {
    fn execute(
        &self,
        sess: &SymbolicSession,
        op: &Operator,
        plc: &Placement,
        operands: Operands<SymbolicValue>,
    ) -> Result<SymbolicValue> {
        use Operator::*;
        match op {
            DeriveSeed(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Constant(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Cast(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Input(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Output(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Load(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Save(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Shape(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Broadcast(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Fill(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            PrfKeyGen(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Decrypt(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Xor(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            And(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Or(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            BitExtract(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Sample(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            SampleSeeded(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            RingFixedpointArgmax(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            RingFixedpointEncode(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            RingFixedpointDecode(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            RingFixedpointMean(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            RingInject(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Share(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Reveal(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            TruncPr(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            AddN(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Shl(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Shr(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Msb(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Abs(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Mux(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Maximum(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Softmax(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Argmax(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Demirror(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Mirror(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            RepToAdt(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Index(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Diag(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            BitDecompose(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            BitCompose(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            ShlDim(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            AdtToRep(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Sqrt(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Sign(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Pow2(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Exp(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Sigmoid(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Log2(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Log(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Equal(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            EqualZero(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            LessThan(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            GreaterThan(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            FixedpointEncode(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            FixedpointDecode(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Identity(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            AtLeast2D(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            IndexAxis(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Slice(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Ones(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            ExpandDims(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Concat(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Reshape(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Squeeze(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Transpose(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Inverse(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Add(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Sub(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Mul(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Div(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Dot(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Mean(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Neg(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Sum(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Send(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Receive(op) => DispatchKernel::compile(op, plc)?(sess, operands),
        }
    }
}

/// Helper for execution computations symbolically.
#[derive(Default)]
pub struct SymbolicExecutor {
    // Placeholder for the future state we want to keep (symbolic strategy pointer, replicated setup cache, etc).
}

impl SymbolicExecutor {
    pub fn run_computation(&self, computation: &Computation) -> anyhow::Result<Computation> {
        let session = SymbolicSession::default();
        let computation = computation.toposort()?;

        {
            let mut env: HashMap<&String, SymbolicValue> =
                HashMap::with_capacity(computation.operations.len());

            for op in computation.operations.iter() {
                let operands = op
                    .inputs
                    .iter()
                    .map(|input_name| env.get(input_name).unwrap().clone())
                    .collect();
                let result = session
                    .execute(&op.kind, &op.placement, operands)
                    .map_err(|e| {
                        Error::Compilation(format!(
                            "SymbolicSession failed to lower computation due to an error: {:?}",
                            e,
                        ))
                    })?;
                env.insert(&op.name, result);
            }
        }

        let state = Arc::try_unwrap(session.state)
            .map_err(|_| Error::Compilation(format!("could not consume state after lowering")))?
            .into_inner();
        let operations = state.ops;
        Ok(Computation { operations })
    }
}
