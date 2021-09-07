use crate::computation::{
    Computation, KnownType, Operation, Operator, Placed, Placement, ReplicatedPlacement,
    SymbolicValue,
};
use crate::error::Error;
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

    fn placement(&self) -> crate::error::Result<Self::Placement> {
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
    fn execute(&self, op: Operator, plc: &Placement, operands: Vec<Self::Value>) -> Self::Value {
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
    ) -> SymbolicValue;
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
    ) -> SymbolicValue {
        use Operator::*;
        match op {
            PrimDeriveSeed(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            Constant(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            Cast(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            Input(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            Output(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            Load(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            Save(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            Shape(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            BitFill(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RingFill(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RepFill(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            PrimPrfKeyGen(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            BitXor(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            BitAnd(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            BitExtract(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            BitSample(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            BitSampleSeeded(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RingSample(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RingSampleSeeded(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RingAdd(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RingSub(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RingMul(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RingDot(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RingNeg(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RingSum(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RingFixedpointEncode(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RingFixedpointDecode(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RingFixedpointMean(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RingShl(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RingShr(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RingInject(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RepSetup(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RepShare(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RepReveal(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RepTruncPr(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RepAdd(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RepSub(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RepMul(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RepDot(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RepFixedpointMean(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RepSum(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RepShl(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RepMsb(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RepAbs(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RepToAdt(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RepIndexAxis(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RepDiag(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RepSlice(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RepBitDec(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            RepShlDim(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            AdtAdd(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            AdtSub(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            AdtShl(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            AdtMul(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            AdtFill(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            AdtReveal(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            AdtToRep(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            HostAtLeast2D(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            HostMean(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            HostSqrt(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            HostSum(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            HostSlice(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            HostDiag(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            HostShlDim(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            HostIndexAxis(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            HostOnes(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            HostAdd(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            HostSub(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            HostMul(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            HostDiv(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            HostDot(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            HostExpandDims(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            HostSqueeze(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            HostConcat(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            HostTranspose(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            HostInverse(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            HostBitDec(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            FixedpointEncode(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            FixedpointDecode(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            FixedpointAdd(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            FixedpointSub(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            FixedpointMul(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            FixedpointDot(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            FixedpointTruncPr(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            FixedpointSum(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            FixedpointMean(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            FloatingpointAdd(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            FloatingpointSub(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            FloatingpointMul(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            FloatingpointDiv(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            FloatingpointDot(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            FloatingpointOnes(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            Identity(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            HostReshape(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            AtLeast2D(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            Slice(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            Ones(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            ExpandDims(op) => unimplemented!("Not done yet: {:?}", op),
            Concat(op) => unimplemented!("Not done yet: {:?}", op),
            Transpose(op) => unimplemented!("Not done yet: {:?}", op),
            Inverse(op) => unimplemented!("Not done yet: {:?}", op),
            Add(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            Sub(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            Mul(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            Div(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            Dot(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            Mean(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            Sum(op) => DispatchKernel::compile(&op, plc)(sess, operands),
            // the following operators are not supported by design (for now, at least)
            Send(_) | Receive(_) => {
                unimplemented!("Unsupported symbolic operator {:?}", op)
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
            let res = session.execute(operator, &op.placement, operands);
            env.insert(op.name.clone(), res);
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
