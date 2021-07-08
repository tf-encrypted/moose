use crate::computation::{
    Operation, Operator, Placed, Placement, ReplicatedPlacement, SymbolicValue,
};
use crate::kernels::Session;
use crate::prim::PrfKey;
use crate::replicated::{AbstractReplicatedSetup, AbstractReplicatedTensor};
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

pub struct SymbolicSession {
    strategy: Box<dyn SymbolicStrategy>,
    ops: Arc<RwLock<Vec<Operation>>>, // TODO use HashMap so we can do some consistency checks on the fly?
    replicated_keys:
        HashMap<ReplicatedPlacement, Symbolic<AbstractReplicatedSetup<Symbolic<PrfKey>>>>,
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
        unimplemented!()
    }

    type ReplicatedSetup = (); // TODO
    fn replicated_setup(&self, plc: &ReplicatedPlacement) -> &Self::ReplicatedSetup {
        unimplemented!()
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
        match op {
            _ => unimplemented!(),
        }
    }
}
