use crate::computation::{
    Computation, HostPlacement, KnownType, Operation, Operator, Placed, Placement,
    ReplicatedPlacement, SymbolicValue,
};
use crate::error::{Error, Result};
use crate::kernels::{DispatchKernel, PlacementPlace, Session};
use crate::prim::PrfKey;
use crate::replicated::AbstractReplicatedSetup;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Clone, Debug, PartialEq)]
pub enum Symbolic<T: Placed> {
    Symbolic(SymbolicHandle<T::Placement>),
    Concrete(T),
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

pub struct SymbolicSession {
    pub strategy: Box<dyn SymbolicStrategy>,
    pub ops: Arc<RwLock<Vec<Operation>>>, // TODO use HashMap so we can do some consistency checks on the fly?
    pub replicated_keys:
        HashMap<ReplicatedPlacement, Symbolic<AbstractReplicatedSetup<Symbolic<PrfKey>>>>,
}

impl Default for SymbolicSession {
    fn default() -> Self {
        SymbolicSession {
            strategy: Box::new(DefaultSymbolicStrategy),
            ops: Default::default(),
            replicated_keys: Default::default(),
        }
    }
}

impl SymbolicSession {
    pub fn add_operation<'s, O: Into<Operator> + Clone>(
        &'s self,
        operator: &O,
        operands: &[&str],
        plc: &Placement,
    ) -> String {
        let mut ops = self.ops.write().unwrap();
        let op_name: String = format!("op_{}", ops.len());
        let op = Operation {
            name: op_name.clone(),
            kind: operator.clone().into(),
            inputs: operands.iter().map(|op| op.to_string()).collect(),
            placement: plc.clone(),
        };
        ops.push(op);
        op_name
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

    type ReplicatedSetup = <crate::replicated::ReplicatedSetup as KnownType<SymbolicSession>>::Type;
    fn replicated_setup(&self, plc: &ReplicatedPlacement) -> &Self::ReplicatedSetup {
        self.replicated_keys.get(plc).unwrap()
    }
}

pub trait SymbolicStrategy {
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
            PrimDeriveSeed(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            Constant(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            Cast(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            Input(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            Output(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            Load(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            Save(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            Shape(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            BitFill(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RingFill(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RepFill(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            PrimPrfKeyGen(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            BitXor(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            BitAnd(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            BitExtract(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            BitSample(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            BitSampleSeeded(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RingSample(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RingSampleSeeded(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RingAdd(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RingSub(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RingMul(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RingDot(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RingNeg(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RingSum(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RingFixedpointEncode(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RingFixedpointDecode(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RingFixedpointMean(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RingShl(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RingShr(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RingInject(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RepSetup(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RepShare(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RepReveal(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RepTruncPr(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RepAdd(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RepSub(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RepMul(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RepDot(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RepMean(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RepSum(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RepShl(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RepMsb(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RepAbs(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RepToAdt(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RepIndexAxis(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RepDiag(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RepSlice(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RepBitDec(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            RepShlDim(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            AdtAdd(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            AdtSub(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            AdtShl(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            AdtMul(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            AdtFill(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            AdtReveal(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            AdtToRep(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            HostAtLeast2D(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            HostMean(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            HostSqrt(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            HostSum(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            HostSlice(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            HostDiag(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            HostShlDim(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            HostIndexAxis(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            HostOnes(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            HostAdd(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            HostSub(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            HostMul(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            HostDiv(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            HostDot(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            HostExpandDims(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            HostSqueeze(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            HostConcat(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            HostTranspose(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            HostInverse(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            HostBitDec(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            FixedpointEncode(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            FixedpointDecode(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            FixedpointAdd(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            FixedpointSub(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            FixedpointMul(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            FixedpointDot(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            FixedpointTruncPr(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            FixedpointSum(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            FixedpointMean(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            Identity(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            HostReshape(op) => Ok(DispatchKernel::compile(&op, plc)?(sess, operands)),
            // the following operators are not supported by design (for now, at least)
            Send(_) | Receive(_) => Err(Error::UnimplementedSymbolicOperator(format!("{:?}", op))),
        }
    }
}

impl PlacementPlace<SymbolicSession, Symbolic<String>> for HostPlacement {
    fn place(&self, _sess: &SymbolicSession, x: Symbolic<String>) -> Symbolic<String> {
        match x {
            Symbolic::Concrete(x) => Symbolic::Concrete(x),
            Symbolic::Symbolic(SymbolicHandle { op, plc: _ }) => {
                Symbolic::Symbolic(SymbolicHandle {
                    op,
                    plc: Placement::Host(self.clone()),
                })
            }
        }
    }
}

pub struct SymbolicExecutor {
    // Placeholder for the future state we want to keep (symbolic strategy pointer, replicated setup cache, etc).
}

impl Default for SymbolicExecutor {
    fn default() -> Self {
        SymbolicExecutor {}
    }
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
        let ops = session.ops.read().map_err(|e| {
            Error::Compilation(format!(
                "Failed to get operations from the Symbolic Session due to an error: {}",
                e
            ))
        })?;
        Ok(Computation {
            operations: ops.clone(),
        })
    }
}
