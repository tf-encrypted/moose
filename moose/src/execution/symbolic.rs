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
use crate::kernels::{NgDispatchKernel, NgKernel};
use crate::replicated::{RepSetup, ReplicatedPlacement};
use crate::{MirroredCounterpart, Ring, TensorLike, Underlying};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Wrapper for values used in `SymbolicSession`s
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
        O: Clone,
        Operator: From<O>,
        P: Clone + Into<Q>,
        Placement: From<P>,
    {
        let mut state = self.state.write();
        let op_name: String = format!("op_{}", state.ops.len());
        let op = Operation {
            name: op_name.clone(),
            kind: Operator::from(operator.clone()),
            inputs: operands.iter().map(|op| op.to_string()).collect(),
            placement: Placement::from(plc.clone()),
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

// pub(crate) fn ternary_symbolic_kernel<U: Placed>(sess: &SymbolicSession, op: &Operator, plc: &Placement, x0: SymbolicValue, x1: SymbolicValue, x2: SymbolicValue) -> SymbolicValue {
//     let h0 = x0.symbolic_handle().unwrap();
//     let h1 = x1.symbolic_handle().unwrap();
//     let h2 = x2.symbolic_handle().unwrap();

//     let h = sess.add_operation(&op.clone(), &[&h0.op, &h1.op, &h2.op], plc);
//     Ok(SymbolicValue::from(Symbolic::<U>::Symbolic(h)))
// }

impl SymbolicStrategy for DefaultSymbolicStrategy {
    fn execute(
        &self,
        sess: &SymbolicSession,
        op: &Operator,
        plc: &Placement,
        mut operands: Operands<SymbolicValue>,
    ) -> Result<SymbolicValue> {
        use Operator::*;
        match op {
            DeriveSeed(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Constant(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Cast(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Input(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Output(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Load(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Save(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Shape(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Broadcast(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Binary { closure } => {
                        assert_eq!(operands.len(), 2);
                        let x1 = operands.pop().unwrap();
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0, x1)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Fill(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            PrfKeyGen(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Nullary { closure } => closure(sess, plc),
                    _ => {
                        return Err(Error::Compilation(
                            "PrfKeyGen should be an unary kernel".to_string(),
                        ))
                    }
                }
            }
            Decrypt(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Binary { closure } => {
                        assert_eq!(operands.len(), 2);
                        let x1 = operands.pop().unwrap();
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0, x1)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Xor(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Binary { closure } => {
                        assert_eq!(operands.len(), 2);
                        let x1 = operands.pop().unwrap();
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0, x1)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            And(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Binary { closure } => {
                        assert_eq!(operands.len(), 2);
                        let x1 = operands.pop().unwrap();
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0, x1)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Or(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Binary { closure } => {
                        assert_eq!(operands.len(), 2);
                        let x1 = operands.pop().unwrap();
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0, x1)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            BitExtract(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0: SymbolicValue = operands.pop().unwrap();

                        let y = closure(sess, plc, x0);
                        y
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Sample(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            SampleSeeded(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Binary { closure } => {
                        assert_eq!(operands.len(), 2);
                        let x1 = operands.pop().unwrap();
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0, x1)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            RingFixedpointAbs(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            RingFixedpointArgmax(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            RingFixedpointEncode(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            RingFixedpointDecode(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            RingFixedpointMean(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            RingInject(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Share(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Reveal(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            TruncPr(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            AddN(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Shl(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Shr(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Msb(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Abs(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Mux(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Ternary { closure } => {
                        assert_eq!(operands.len(), 3);
                        let x2: SymbolicValue = operands.pop().unwrap();
                        let x1: SymbolicValue = operands.pop().unwrap();
                        let x0: SymbolicValue = operands.pop().unwrap();

                        let y = closure(sess, plc, x0, x1, x2);
                        y
                    }
                    _ => Err(Error::Compilation(
                        "MuxOp should be a ternary kernel".to_string(),
                    )),
                }
            }
            Maximum(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Softmax(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Argmax(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Demirror(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Mirror(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            RepToAdt(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Index(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Diag(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            BitDecompose(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            BitCompose(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            ShlDim(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            AdtToRep(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Sqrt(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Sign(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Pow2(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Exp(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Sigmoid(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Log2(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Log(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Equal(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Binary { closure } => {
                        assert_eq!(operands.len(), 2);
                        let x1 = operands.pop().unwrap();
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0, x1)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            EqualZero(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            LessThan(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Binary { closure } => {
                        assert_eq!(operands.len(), 2);
                        let x1 = operands.pop().unwrap();
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0, x1)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            GreaterThan(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Binary { closure } => {
                        assert_eq!(operands.len(), 2);
                        let x1 = operands.pop().unwrap();
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0, x1)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            FixedpointEncode(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            FixedpointDecode(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Identity(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            AtLeast2D(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            IndexAxis(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Slice(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Ones(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            ExpandDims(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Concat(op) => DispatchKernel::compile(op, plc)?(sess, operands),
            Reshape(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Binary { closure } => {
                        assert_eq!(operands.len(), 2);
                        let x1 = operands.pop().unwrap();
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0, x1)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Squeeze(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Transpose(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Inverse(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Add(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Binary { closure } => {
                        assert_eq!(operands.len(), 2);
                        let x1 = operands.pop().unwrap();
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0, x1)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Sub(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Binary { closure } => {
                        assert_eq!(operands.len(), 2);
                        let x1 = operands.pop().unwrap();
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0, x1)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Mul(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Binary { closure } => {
                        assert_eq!(operands.len(), 2);
                        let x1 = operands.pop().unwrap();
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0, x1)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Div(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Binary { closure } => {
                        assert_eq!(operands.len(), 2);
                        let x1 = operands.pop().unwrap();
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0, x1)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Dot(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Binary { closure } => {
                        assert_eq!(operands.len(), 2);
                        let x1 = operands.pop().unwrap();
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0, x1)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Mean(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Neg(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
            Sum(op) => {
                let kernel = NgDispatchKernel::compile(op, plc)?;
                match kernel {
                    NgKernel::Unary { closure } => {
                        assert_eq!(operands.len(), 1);
                        let x0 = operands.pop().unwrap();
                        closure(sess, plc, x0)
                    }
                    _ => {
                        return Err(Error::Compilation(format!(
                            "Should have gotten an unary kernel for {:?}",
                            op
                        )))
                    }
                }
            }
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
                            "SymbolicSession failed to lower computation due to an error: {}",
                            e,
                        ))
                    })?;
                env.insert(&op.name, result);
            }
        }

        let state = Arc::try_unwrap(session.state)
            .map_err(|_| Error::Compilation("could not consume state after lowering".to_string()))?
            .into_inner();
        let operations = state.ops;
        Ok(Computation { operations })
    }
}
