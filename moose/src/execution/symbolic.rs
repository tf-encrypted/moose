//! Symbolic execution of computations
//!
//! This is used during compilation to lower operations.
//! In general, it works by evaluating kernels on symbolic values and
//! recording the underlying operations perform as new computation.

use crate::computation::{
    Computation, KnownType, Operation, Operator, Placed, Placement, SymbolicValue,
};
use crate::error::{Error, Result};
use crate::execution::Session;
use crate::kernels::{DispatchKernel, PlacementPlace};
use crate::replicated::ReplicatedPlacement;
use crate::types::ReplicatedSetup;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub enum Symbolic<T: Placed> {
    Symbolic(SymbolicHandle<T::Placement>),
    Concrete(T),
}

impl<T: Placed> Symbolic<T> {
    pub fn is_symbolic(&self) -> bool {
        match self {
            Symbolic::Symbolic(_) => true,
            Symbolic::Concrete(_) => false,
        }
    }

    pub fn is_concrete(&self) -> bool {
        match self {
            Symbolic::Symbolic(_) => false,
            Symbolic::Concrete(_) => true,
        }
    }

    pub fn symbolic_handle(&self) -> Option<&SymbolicHandle<T::Placement>> {
        match self {
            Symbolic::Symbolic(h) => Some(h),
            Symbolic::Concrete(_) => None,
        }
    }
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
    pub replicated_keys:
        HashMap<ReplicatedPlacement, Arc<<ReplicatedSetup as KnownType<SymbolicSession>>::Type>>,
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
    pub fn add_operation<'s, O: Into<Operator> + Clone, P: Into<Placement> + Clone>(
        &'s self,
        operator: &O,
        operands: &[&str],
        plc: &P,
    ) -> SymbolicHandle<P> {
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
            plc: plc.clone(),
        }
    }

    /// Apply a given closure to the iterator over the ops.
    ///
    /// The "ops" vector is locked for READ for the duration of the call.
    pub fn ops_iter<F: FnMut(std::slice::Iter<Operation>) -> T, T>(&self, mut operation: F) -> T {
        let state = self.state.read();
        operation(state.ops.iter())
    }
}

impl Session for SymbolicSession {
    type Value = crate::computation::SymbolicValue;
    fn execute(
        &self,
        op: Operator,
        plc: &Placement,
        operands: Vec<Self::Value>,
    ) -> Result<Self::Value> {
        self.strategy.execute(self, op, plc, operands)
    }

    type ReplicatedSetup = <ReplicatedSetup as KnownType<SymbolicSession>>::Type;

    fn replicated_setup(&self, plc: &ReplicatedPlacement) -> Arc<Self::ReplicatedSetup> {
        // Produce a new replicated setup or returned a previously produced setup for the placement
        let state = self.state.read();
        match state.replicated_keys.get(plc) {
            Some(setup) => Arc::clone(setup),
            None => {
                use crate::kernels::PlacementSetupGen;
                drop(state); // Release the read access

                // This may (likely) grab a write lock to the state inside
                let new_setup = plc.gen_setup(self);

                // Grab a new write lock.
                let mut state = self.state.write();
                // Only insert if missing, since someone else might have done that already
                // If our `new_setup` ends up being unused due to the race, it will be pruned later on by a dedicated pruning pass.
                let setup = state
                    .replicated_keys
                    .entry(plc.clone())
                    .or_insert_with(|| Arc::new(new_setup));
                Arc::clone(setup)
            }
        }
    }
}

pub(crate) trait SymbolicStrategy {
    fn execute(
        &self,
        sess: &SymbolicSession,
        op: Operator,
        plc: &Placement,
        operands: Vec<SymbolicValue>,
    ) -> Result<SymbolicValue>;
}

#[derive(Clone, Copy, Debug)]
struct DefaultSymbolicStrategy;

impl SymbolicStrategy for DefaultSymbolicStrategy {
    fn execute(
        &self,
        sess: &SymbolicSession,
        op: Operator,
        plc: &Placement,
        operands: Vec<SymbolicValue>,
    ) -> Result<SymbolicValue> {
        use Operator::*;
        match op {
            PrimDeriveSeed(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Constant(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Cast(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Input(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Output(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Load(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Save(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Shape(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Broadcast(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            RingFill(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Fill(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            PrimPrfKeyGen(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            AesDecrypt(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            BitXor(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            BitAnd(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Neg(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            BitOr(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            BitExtract(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            BitSample(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            BitSampleSeeded(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            RingSample(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            RingSampleSeeded(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            RingFixedpointEncode(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            RingFixedpointDecode(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            RingFixedpointMean(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            RingInject(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            RepSetup(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            RepShare(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            RepReveal(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            RepTruncPr(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            RepFixedpointMean(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            AddN(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Shl(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Shr(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            RepMsb(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            RepAbs(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            RepAnd(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            RepXor(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            RepEqual(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Mux(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Maximum(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Softmax(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Demirror(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Mirror(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            RepToAdt(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Index(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            RepDiag(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            RepSlice(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            RepBitDec(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            RepBitCompose(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            RepShlDim(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            AdtFill(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            AdtReveal(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            AdtToRep(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            HostAtLeast2D(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            HostMean(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Sqrt(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            HostSlice(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            HostDiag(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            HostShlDim(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            HostOnes(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Sign(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Pow2(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Exp(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Sigmoid(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Less(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            GreaterThan(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            HostSqueeze(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            HostTranspose(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            HostInverse(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            HostBitDec(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            FixedpointEncode(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            FixedpointDecode(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            FixedpointTruncPr(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            FixedpointMean(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Identity(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            HostReshape(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            FloatingpointDiv(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            FloatingpointAtLeast2D(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            FloatingpointOnes(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            FloatingpointConcat(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            FloatingpointTranspose(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            FloatingpointInverse(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            FloatingpointMean(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            AtLeast2D(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            IndexAxis(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Slice(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Ones(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            ExpandDims(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Concat(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Transpose(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Inverse(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Add(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Sub(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Mul(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Div(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Dot(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Mean(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Neg(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Sum(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Send(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
            Receive(op) => DispatchKernel::compile(&op, plc)?(sess, operands),
        }
    }
}

/// Helper for execution computations symbolically.
#[derive(Default)]
pub struct SymbolicExecutor {
    // Placeholder for the future state we want to keep (symbolic strategy pointer, replicated setup cache, etc).
}

impl SymbolicExecutor {
    pub fn run_computation(
        &self,
        computation: &Computation,
        session: &SymbolicSession,
    ) -> anyhow::Result<Computation> {
        let mut env: HashMap<String, SymbolicValue> = HashMap::default();
        let computation = computation.toposort()?;

        for op in computation.operations.iter() {
            let operator = op.kind.clone();
            let operands = op
                .inputs
                .iter()
                .map(|input_name| env.get(input_name).unwrap().clone())
                .collect();
            let value = session
                .execute(operator, &op.placement, operands)
                .map_err(|e| {
                    Error::Compilation(format!(
                        "SymbolicSession failed to lower computation due to an error: {:?}",
                        e,
                    ))
                })?;
            env.insert(op.name.clone(), value);
        }
        let state = session.state.read();
        Ok(Computation {
            operations: state.ops.clone(),
        })
    }
}
