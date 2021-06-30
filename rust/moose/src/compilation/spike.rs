#![allow(dead_code)]
#![allow(unused_variables)]

use macros::with_context;
use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::ops::{Add, Mul, Shl, Shr, Sub};
use std::ops::{BitAnd, BitXor};

impl Ty {
    pub fn synthesize_symbolic_value<S: Into<String>>(
        &self,
        op_name: S,
        plc: Placement,
    ) -> SymbolicValue {
        match self {
            Ty::Fixed64Tensor => SymbolicValue::Fixed64Tensor(Symbolic::Symbolic(SymbolicHandle {
                op: op_name.into(),
                plc,
            })),
            Ty::Fixed128Tensor => {
                SymbolicValue::Fixed128Tensor(Symbolic::Symbolic(SymbolicHandle {
                    op: op_name.into(),
                    plc,
                }))
            }
            Ty::BitTensor => SymbolicValue::BitTensor(Symbolic::Symbolic(SymbolicHandle {
                op: op_name.into(),
                plc: plc.try_into().unwrap(),
            })),
            Ty::Ring64Tensor => SymbolicValue::Ring64Tensor(Symbolic::Symbolic(SymbolicHandle {
                op: op_name.into(),
                plc: plc.try_into().unwrap(),
            })),
            Ty::Ring128Tensor => SymbolicValue::Ring128Tensor(Symbolic::Symbolic(SymbolicHandle {
                op: op_name.into(),
                plc: plc.try_into().unwrap(),
            })),
            Ty::Replicated64Tensor => {
                SymbolicValue::Replicated64Tensor(Symbolic::Symbolic(SymbolicHandle {
                    op: op_name.into(),
                    plc: plc.try_into().unwrap(),
                }))
            }
            Ty::Replicated128Tensor => {
                SymbolicValue::Replicated128Tensor(Symbolic::Symbolic(SymbolicHandle {
                    op: op_name.into(),
                    plc: plc.try_into().unwrap(),
                }))
            }
            Ty::ReplicatedBitTensor => {
                SymbolicValue::ReplicatedBitTensor(Symbolic::Symbolic(SymbolicHandle {
                    op: op_name.into(),
                    plc: plc.try_into().unwrap(),
                }))
            }
            Ty::Additive64Tensor => {
                SymbolicValue::Additive64Tensor(Symbolic::Symbolic(SymbolicHandle {
                    op: op_name.into(),
                    plc: plc.try_into().unwrap(),
                }))
            }
            Ty::Additive128Tensor => {
                SymbolicValue::Additive128Tensor(Symbolic::Symbolic(SymbolicHandle {
                    op: op_name.into(),
                    plc: plc.try_into().unwrap(),
                }))
            }
            Ty::ReplicatedSetup => {
                SymbolicValue::ReplicatedSetup(Symbolic::Symbolic(SymbolicHandle {
                    op: op_name.into(),
                    plc: plc.try_into().unwrap(),
                }))
            }
            Ty::PrfKey => SymbolicValue::PrfKey(Symbolic::Symbolic(SymbolicHandle {
                op: op_name.into(),
                plc: plc.try_into().unwrap(),
            })),
        }
    }
}

pub trait KnownType {
    type Symbolic;
    // const TY: Ty;
}

#[derive(Clone, Debug, PartialEq)]
pub enum SymbolicValue {
    Fixed64Tensor(<Fixed64Tensor as KnownType>::Symbolic),
    Fixed128Tensor(<Fixed128Tensor as KnownType>::Symbolic),
    BitTensor(<BitTensor as KnownType>::Symbolic),
    Ring64Tensor(<Ring64Tensor as KnownType>::Symbolic),
    Ring128Tensor(<Ring128Tensor as KnownType>::Symbolic),
    Replicated64Tensor(<Replicated64Tensor as KnownType>::Symbolic),
    Replicated128Tensor(<Replicated128Tensor as KnownType>::Symbolic),
    ReplicatedBitTensor(<ReplicatedBitTensor as KnownType>::Symbolic),
    Additive64Tensor(<Additive64Tensor as KnownType>::Symbolic),
    Additive128Tensor(<Additive128Tensor as KnownType>::Symbolic),
    ReplicatedSetup(<ReplicatedSetup as KnownType>::Symbolic),
    PrfKey(<PrfKey as KnownType>::Symbolic),
}

macro_rules! value {
    ($t:ident, $st:ty) => {
        // impl From<$t> for Value {
        //     fn from(x: $t) -> Value {
        //         Value::$t(x)
        //     }
        // }

        // impl From<&$t> for Value {
        //     fn from(x: &$t) -> Value {
        //         Value::$t(x.clone())
        //     }
        // }

        // impl TryFrom<Value> for $t {
        //     type Error = ();

        //     fn try_from(x: Value) -> Result<Self, Self::Error> {
        //         match x {
        //             Value::$t(x) => Ok(x),
        //             _ => Err(()),
        //         }
        //     }
        // }

        impl From<<$t as KnownType>::Symbolic> for SymbolicValue {
            fn from(x: <$t as KnownType>::Symbolic) -> SymbolicValue {
                SymbolicValue::$t(x)
            }
        }

        impl TryFrom<SymbolicValue> for <$t as KnownType>::Symbolic {
            type Error = ();

            fn try_from(x: SymbolicValue) -> Result<Self, Self::Error> {
                match x {
                    SymbolicValue::$t(x) => Ok(x),
                    _ => Err(()),
                }
            }
        }

        impl KnownType for $t {
            type Symbolic = $st;
            // const TY: Ty = Ty::$t;
        }

        impl KnownType for $st {
            type Symbolic = Self;
            const TY: Ty = Ty::$t;
        }
    };
}

// NOTE a future improvement might be to have a single `values!` macro
// that takes care of everything, including generating `enum Value` and
// `enum SymbolicValue` and maybe even `enum Ty`.
// one thing to be careful about here is to still make room for manual
// constructions during development.
value!(
    // Fixed64Tensor,
    Symbolic<
        FixedTensor<
            <Ring64Tensor as KnownType>::Symbolic,
            <Replicated64Tensor as KnownType>::Symbolic,
        >,
    >
);
value!(
    // Fixed128Tensor,
    Symbolic<
        FixedTensor<
            <Ring128Tensor as KnownType>::Symbolic,
            <Replicated128Tensor as KnownType>::Symbolic,
        >,
    >
);
value!(
    // BitTensor,
    Symbolic<BitTensor>,
);
value!(
    // Ring64Tensor,
    Symbolic<Ring64Tensor>,
);
value!(
    // Ring128Tensor,
    Symbolic<Ring128Tensor>,
);
value!(
    // Replicated64Tensor,
    Symbolic<ReplicatedTensor<<Ring64Tensor as KnownType>::Symbolic>>
);
value!(
    // Replicated128Tensor,
    Symbolic<ReplicatedTensor<<Ring128Tensor as KnownType>::Symbolic>>
);
value!(
    // ReplicatedBitTensor,
    Symbolic<ReplicatedTensor<Symbolic<BitTensor>>>
);
value!(
    // Additive64Tensor,
    Symbolic<AdditiveTensor<<Ring64Tensor as KnownType>::Symbolic>>
);
value!(
    // Additive128Tensor,
    Symbolic<AdditiveTensor<<Ring128Tensor as KnownType>::Symbolic>>
);
value!(
    // ReplicatedSetup,
    Symbolic<AbstractReplicatedSetup<<PrfKey as KnownType>::Symbolic>>
);
value!(
    // PrfKey,
    Symbolic<PrfKey>,
);

#[derive(Clone, Debug, PartialEq)]
pub enum Symbolic<T: Placed> {
    Symbolic(SymbolicHandle<T::Placement>),
    Concrete(T),
}

impl<T: Placed> Placed for Symbolic<T>
where
    T::Placement: Clone,
{
    type Placement = T::Placement;

    fn placement(&self) -> Self::Placement {
        match self {
            Symbolic::Symbolic(x) => x.plc.clone(),
            Symbolic::Concrete(x) => x.placement(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SymbolicHandle<P> {
    op: String,
    // NOTE if we had a handle to the graph we
    // could perhaps derive the placement instead
    plc: P,
}

impl<T: Placed> From<SymbolicHandle<T::Placement>> for Symbolic<T> {
    fn from(x: SymbolicHandle<T::Placement>) -> Symbolic<T> {
        Symbolic::Symbolic(x)
    }
}

impl<K> TryFrom<Symbolic<AbstractReplicatedSetup<K>>> for AbstractReplicatedSetup<K>
where
    K: Placed<Placement = HostPlacement>,
{
    type Error = Symbolic<Self>;

    fn try_from(x: Symbolic<AbstractReplicatedSetup<K>>) -> Result<Self, Self::Error> {
        match x {
            Symbolic::Concrete(cx) => Ok(cx),
            Symbolic::Symbolic(_) => Err(x),
        }
    }
}

impl<R> TryFrom<Symbolic<ReplicatedTensor<R>>> for ReplicatedTensor<R>
where
    R: Placed<Placement = HostPlacement>,
{
    type Error = Symbolic<Self>;

    fn try_from(x: Symbolic<ReplicatedTensor<R>>) -> Result<Self, Self::Error> {
        match x {
            Symbolic::Concrete(cx) => Ok(cx),
            Symbolic::Symbolic(_) => Err(x),
        }
    }
}

impl<R> TryFrom<Symbolic<AdditiveTensor<R>>> for AdditiveTensor<R>
where
    R: Placed<Placement = HostPlacement>,
{
    type Error = Symbolic<Self>;

    fn try_from(x: Symbolic<AdditiveTensor<R>>) -> Result<Self, Self::Error> {
        match x {
            Symbolic::Concrete(cx) => Ok(cx),
            Symbolic::Symbolic(_) => Err(x),
        }
    }
}

impl<RingTensorT, ReplicatedTensorT> TryFrom<Symbolic<FixedTensor<RingTensorT, ReplicatedTensorT>>>
    for FixedTensor<RingTensorT, ReplicatedTensorT>
where
    RingTensorT: Placed<Placement = HostPlacement>,
    ReplicatedTensorT: Placed<Placement = ReplicatedPlacement>,
{
    type Error = Symbolic<Self>;

    fn try_from(
        x: Symbolic<FixedTensor<RingTensorT, ReplicatedTensorT>>,
    ) -> Result<Self, Self::Error> {
        match x {
            Symbolic::Concrete(cx) => Ok(cx),
            Symbolic::Symbolic(_) => Err(x),
        }
    }
}

impl<RingTensorT, ReplicatedTensorT> From<FixedTensor<RingTensorT, ReplicatedTensorT>>
    for Symbolic<FixedTensor<RingTensorT, ReplicatedTensorT>>
where
    RingTensorT: Placed<Placement = HostPlacement>,
    ReplicatedTensorT: Placed<Placement = ReplicatedPlacement>,
{
    fn from(x: FixedTensor<RingTensorT, ReplicatedTensorT>) -> Self {
        Symbolic::Concrete(x)
    }
}

impl<T> From<RingTensor<T>> for Symbolic<RingTensor<T>> {
    fn from(x: RingTensor<T>) -> Self {
        Symbolic::Concrete(x)
    }
}

impl<R> From<ReplicatedTensor<R>> for Symbolic<ReplicatedTensor<R>>
where
    R: Placed<Placement = HostPlacement>,
{
    fn from(x: ReplicatedTensor<R>) -> Self {
        Symbolic::Concrete(x)
    }
}

impl<K> From<AbstractReplicatedSetup<K>> for Symbolic<AbstractReplicatedSetup<K>>
where
    K: Placed<Placement = HostPlacement>,
{
    fn from(x: AbstractReplicatedSetup<K>) -> Self {
        Symbolic::Concrete(x)
    }
}

impl<R> From<AdditiveTensor<R>> for Symbolic<AdditiveTensor<R>>
where
    R: Placed<Placement = HostPlacement>,
{
    fn from(x: AdditiveTensor<R>) -> Self {
        Symbolic::Concrete(x)
    }
}

use crate::computation::{Signature, NullarySignature, UnarySignature, BinarySignature, TernarySignature};

macro_rules! modelled {
    /*
    Nullary
    */
    ($t:ident::$f:ident, $plc:ty, $(attributes[$($attr_id:ident : $attr_ty:ty),*])? () -> $u:ty, $op:ident) => {
        impl NullaryKernelCheck<ConcreteContext, $plc, $u> for $op {}

        impl $t<ConcreteContext, $u> for $plc {
            fn $f(&self, ctx: &ConcreteContext, $($($attr_id:$attr_ty),*)?) -> $u {
                let sig = NullarySignature {
                    ret: <$u as KnownType>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };
                ctx.execute(op.into(), &self.into(), vec![])
                    .try_into()
                    .unwrap()
            }
        }

        impl $t<SymbolicContext, <$u as KnownType>::Symbolic> for $plc {
            fn $f(&self, ctx: &SymbolicContext, $($($attr_id:$attr_ty),*)?) -> <$u as KnownType>::Symbolic {
                let sig = NullarySignature {
                    ret: <$u as KnownType>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };
                ctx.execute(op.into(), &self.into(), vec![])
                    .try_into()
                    .unwrap()
            }
        }
    };

    /*
    Unary
    */
    ($t:ident::$f:ident, $plc:ty, $(attributes[$($attr_id:ident : $attr_ty:ty),*])? ($t0:ty) -> $u:ty, $op:ident) => {
        impl UnaryKernelCheck<ConcreteContext, $plc, $t0, $u> for $op {}

        impl $t<ConcreteContext, $t0> for $plc {
            type Output = $u;

            fn $f(&self, ctx: &ConcreteContext, $($($attr_id:$attr_ty),*,)? x0: &$t0) -> Self::Output {
                let sig = UnarySignature {
                    arg0: <$t0 as KnownType>::TY,
                    ret: <$u as KnownType>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };
                ctx.execute(op.into(), &self.into(), vec![x0.clone().into()])
                    .try_into()
                    .unwrap()
            }
        }

        impl $t<SymbolicContext, <$t0 as KnownType>::Symbolic> for $plc {
            type Output = <$u as KnownType>::Symbolic;

            fn $f(&self, ctx: &SymbolicContext, $($($attr_id:$attr_ty),*,)? x0: &<$t0 as KnownType>::Symbolic) -> Self::Output {
                let sig = UnarySignature {
                    arg0: <<$t0 as KnownType>::Symbolic as KnownType>::TY,
                    ret: <<$u as KnownType>::Symbolic as KnownType>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };
                ctx.execute(op.into(), &self.into(), vec![x0.clone().into()])
                    .try_into()
                    .unwrap()
            }
        }
    };

    /*
    Binary
    */
    ($t:ident::$f:ident, $plc:ty, $(attributes[$($attr_id:ident : $attr_ty:ty),*])? ($t0:ty, $t1:ty) -> $u:ty, $op:ident) => {
        impl BinaryKernelCheck<ConcreteContext, $plc, $t0, $t1, $u> for $op {}

        impl $t<ConcreteContext, $t0, $t1> for $plc {
            type Output = $u;

            fn $f(&self, ctx: &ConcreteContext, $($($attr_id:$attr_ty),*,)? x0: &$t0, x1: &$t1) -> Self::Output {
                let sig = BinarySignature {
                    arg0: <$t0 as KnownType>::TY,
                    arg1: <$t1 as KnownType>::TY,
                    ret: <$u as KnownType>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };
                ctx.execute(
                    op.into(),
                    &self.into(),
                    vec![x0.clone().into(), x1.clone().into()],
                )
                .try_into()
                .unwrap()
            }
        }

        impl $t<SymbolicContext, <$t0 as KnownType>::Symbolic, <$t1 as KnownType>::Symbolic>
            for $plc
        {
            type Output = <$u as KnownType>::Symbolic;

            fn $f(
                &self,
                ctx: &SymbolicContext,
                $($($attr_id:$attr_ty),*,)?
                x0: &<$t0 as KnownType>::Symbolic,
                x1: &<$t1 as KnownType>::Symbolic,
            ) -> Self::Output {
                let sig = BinarySignature {
                    arg0: <<$t0 as KnownType>::Symbolic as KnownType>::TY,
                    arg1: <<$t1 as KnownType>::Symbolic as KnownType>::TY,
                    ret: <<$u as KnownType>::Symbolic as KnownType>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };
                ctx.execute(
                    op.into(),
                    &self.into(),
                    vec![x0.clone().into(), x1.clone().into()],
                )
                .try_into()
                .unwrap()
            }
        }
    };

    /*
    Ternary
    */
    ($t:ident::$f:ident, $plc:ty, $(attributes[$($attr_id:ident : $attr_ty:ty),*])? ($t0:ty, $t1:ty, $t2:ty) -> $u:ty, $op:ident) => {
        impl TernaryKernelCheck<ConcreteContext, $plc, $t0, $t1, $t2, $u> for $op {}

        impl $t<ConcreteContext, $t0, $t1, $t2> for $plc {
            type Output = $u;

            fn $f(&self, ctx: &ConcreteContext, $($($attr_id:$attr_ty),*,)? x0: &$t0, x1: &$t1, x2: &$t2) -> Self::Output {
                let sig = TernarySignature {
                    arg0: <$t0 as KnownType>::TY,
                    arg1: <$t1 as KnownType>::TY,
                    arg2: <$t2 as KnownType>::TY,
                    ret: <$u as KnownType>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };
                ctx.execute(
                    op.into(),
                    &self.into(),
                    vec![x0.clone().into(), x1.clone().into(), x2.clone().into()],
                )
                .try_into()
                .unwrap()
            }
        }

        impl
            $t<
                SymbolicContext,
                <$t0 as KnownType>::Symbolic,
                <$t1 as KnownType>::Symbolic,
                <$t2 as KnownType>::Symbolic,
            > for $plc
        {
            type Output = <$u as KnownType>::Symbolic;

            fn $f(
                &self,
                ctx: &SymbolicContext,
                $($($attr_id:$attr_ty),*,)?
                x0: &<$t0 as KnownType>::Symbolic,
                x1: &<$t1 as KnownType>::Symbolic,
                x2: &<$t2 as KnownType>::Symbolic,
            ) -> Self::Output {
                let sig = TernarySignature {
                    arg0: <<$t0 as KnownType>::Symbolic as KnownType>::TY,
                    arg1: <<$t1 as KnownType>::Symbolic as KnownType>::TY,
                    arg2: <<$t2 as KnownType>::Symbolic as KnownType>::TY,
                    ret: <<$u as KnownType>::Symbolic as KnownType>::TY,
                };
                let op = $op {
                    sig: sig.into(),
                    $($($attr_id),*)?
                };
                ctx.execute(
                    op.into(),
                    &self.into(),
                    vec![x0.clone().into(), x1.clone().into(), x2.clone().into()],
                )
                .try_into()
                .unwrap()
            }
        }
    };
}

macro_rules! modelled_alias {
    /*
    Binary
    */
    ($src_t:ident::$src_f:ident, $plc:ty, ($t0:ty, $t1:ty) -> $u:ty => $dst_t:ident::$dst_f:ident) => {
        impl $src_t<ConcreteContext, $t0, $t1> for $plc {
            type Output = $u;

            fn $src_f(&self, ctx: &ConcreteContext, x0: &$t0, x1: &$t1) -> Self::Output {
                $dst_t::$dst_f(self, ctx, x0, x1)
            }
        }

        impl $src_t<SymbolicContext, <$t0 as KnownType>::Symbolic, <$t1 as KnownType>::Symbolic>
            for $plc
        {
            type Output = <$u as KnownType>::Symbolic;

            fn $src_f(
                &self,
                ctx: &SymbolicContext,
                x0: &<$t0 as KnownType>::Symbolic,
                x1: &<$t1 as KnownType>::Symbolic,
            ) -> Self::Output {
                $dst_t::$dst_f(self, ctx, x0, x1)
            }
        }
    };
}

impl Default for ConcreteContext {
    fn default() -> Self {
        ConcreteContext {
            replicated_keys: Default::default(),
        }
    }
}

impl Context for ConcreteContext {
    type Value = Value;

    fn execute(&self, op: Operator, plc: &Placement, operands: Vec<Value>) -> Value {
        match op {
            Operator::PrfKeyGenOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::RingSampleOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::BitSampleOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::RingAddOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::BitXorOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::BitAndOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::RingSubOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::RingMulOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::RingShlOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::RingShrOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::RepSetupOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::RepShareOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::RepRevealOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::RepAddOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::RepMulOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::RepToAddOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::AdditiveAddOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::AdditiveMulOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::AdditiveRevealOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::ConstantOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::FixedAddOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::FixedMulOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
        }
    }

    type ReplicatedSetup = ReplicatedSetup;
    fn replicated_setup(&self, plc: &ReplicatedPlacement) -> &Self::ReplicatedSetup {
        self.replicated_keys.get(plc).unwrap()
    }
}

use std::sync::{Arc, RwLock};

pub struct SymbolicContext {
    strategy: Box<dyn SymbolicStrategy>,
    ops: Arc<RwLock<Vec<Operation>>>, // TODO use HashMap so we can do some consistency checks on the fly?
    replicated_keys:
        HashMap<ReplicatedPlacement, Symbolic<AbstractReplicatedSetup<Symbolic<PrfKey>>>>,
}

impl Default for SymbolicContext {
    fn default() -> Self {
        SymbolicContext {
            strategy: Box::new(DefaultSymbolicStrategy),
            ops: Default::default(),
            replicated_keys: Default::default(),
        }
    }
}

impl Context for SymbolicContext {
    type Value = SymbolicValue;

    fn execute(
        &self,
        op: Operator,
        plc: &Placement,
        operands: Vec<SymbolicValue>,
    ) -> SymbolicValue {
        self.strategy.execute(self, op, plc, operands)
    }

    type ReplicatedSetup = <ReplicatedSetup as KnownType>::Symbolic;
    fn replicated_setup(&self, plc: &ReplicatedPlacement) -> &Self::ReplicatedSetup {
        self.replicated_keys.get(plc).unwrap()
    }
}

impl SymbolicContext {
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
            operator: operator.clone().into(),
            operands: operands.iter().map(|op| op.to_string()).collect(),
            plc: plc.clone(),
        };
        ops.push(op);
        op_name
    }
}

trait SymbolicStrategy {
    fn execute(
        &self,
        ctx: &SymbolicContext,
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
        ctx: &SymbolicContext,
        op: Operator,
        plc: &Placement,
        operands: Vec<SymbolicValue>,
    ) -> SymbolicValue {
        match op {
            Operator::PrfKeyGenOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RingSampleOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::BitSampleOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RingAddOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::BitXorOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::BitAndOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RingSubOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RingMulOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RingShlOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RingShrOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RepSetupOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RepShareOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RepRevealOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RepAddOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RepMulOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RepToAddOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::AdditiveAddOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::AdditiveMulOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::AdditiveRevealOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::ConstantOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::FixedAddOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::FixedMulOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
        }
    }
}

macro_rules! symbolic_dispatch_kernel {

    /*
    Nullary
    */

    ($op:ty, [$( ($plc:ty, () -> $u:ty), )+]) => {
        impl DispatchKernel<SymbolicContext> for $op {
            fn compile<'c>(&self, ctx: &'c SymbolicContext, plc: &Placement) -> Box<dyn Fn(Vec<SymbolicValue>) -> SymbolicValue + 'c> {
                match (plc.ty(), self.sig) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Nullary(NullarySignature{
                                ret: <<$u as KnownType>::Symbolic as KnownType>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into().unwrap();

                            let k = <$op as NullaryKernel<
                                SymbolicContext,
                                $plc,
                                <$u as KnownType>::Symbolic,
                            >>::compile(self, &ctx, &plc);

                            Box::new(move |operands| {
                                assert_eq!(operands.len(), 0);

                                let y: <$u as KnownType>::Symbolic = k(&ctx, &plc);
                                y.into()
                            })
                        }
                    )+
                    _ => unimplemented!(), // ok
                }
            }
        }
    };

    /*
    Unary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty) -> $u:ty), )+]) => {
        impl DispatchKernel<SymbolicContext> for $op {
            fn compile<'c>(&self, ctx: &'c SymbolicContext, plc: &Placement) -> Box<dyn Fn(Vec<SymbolicValue>) -> SymbolicValue + 'c> {
                match (plc.ty(), self.sig) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Unary(UnarySignature{
                                arg0: <<$t0 as KnownType>::Symbolic as KnownType>::TY,
                                ret: <<$u as KnownType>::Symbolic as KnownType>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into().unwrap();

                            let k = <$op as UnaryKernel<
                                SymbolicContext,
                                $plc,
                                <$t0 as KnownType>::Symbolic,
                                <$u as KnownType>::Symbolic,
                            >>::compile(self, &ctx, &plc);

                            Box::new(move |operands| {
                                assert_eq!(operands.len(), 1);

                                let x0: <$t0 as KnownType>::Symbolic = operands.get(0).unwrap().clone().try_into().unwrap();

                                let y: <$u as KnownType>::Symbolic = k(&ctx, &plc, x0);
                                y.into()
                            })
                        }
                    )+
                    _ => unimplemented!(), // ok
                }
            }
        }
    };

    /*
    Binary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty, $t1:ty) -> $u:ty), )+]) => {
        impl DispatchKernel<SymbolicContext> for $op {
            fn compile<'c>(
                &self,
                ctx: &'c SymbolicContext,
                plc: &Placement,
            ) -> Box<dyn Fn(Vec<SymbolicValue>) -> SymbolicValue + 'c> {
                match (plc.ty(), self.sig) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Binary(BinarySignature{
                                arg0: <<$t0 as KnownType>::Symbolic as KnownType>::TY,
                                arg1: <<$t1 as KnownType>::Symbolic as KnownType>::TY,
                                ret: <<$u as KnownType>::Symbolic as KnownType>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into().unwrap();

                            let k = <$op as BinaryKernel<
                                SymbolicContext,
                                $plc,
                                <$t0 as KnownType>::Symbolic,
                                <$t1 as KnownType>::Symbolic,
                                <$u as KnownType>::Symbolic,
                            >>::compile(self, &ctx, &plc);

                            Box::new(move |operands| {
                                assert_eq!(operands.len(), 2);

                                let x0: <$t0 as KnownType>::Symbolic = operands.get(0).unwrap().clone().try_into().unwrap();
                                let x1: <$t1 as KnownType>::Symbolic = operands.get(1).unwrap().clone().try_into().unwrap();

                                let y: <$u as KnownType>::Symbolic = k(&ctx, &plc, x0, x1);
                                y.into()
                            })
                        }
                    )+
                    _ => unimplemented!(), // ok
                }
            }

        }
    };

    /*
    Ternary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty), )+]) => {
        impl DispatchKernel<SymbolicContext> for $op {
            fn compile<'c>(
                &self,
                ctx: &'c SymbolicContext,
                plc: &Placement,
            ) -> Box<dyn Fn(Vec<SymbolicValue>) -> SymbolicValue + 'c> {
                match (plc.ty(), self.sig) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Ternary(TernarySignature{
                                arg0: <<$t0 as KnownType>::Symbolic as KnownType>::TY,
                                arg1: <<$t1 as KnownType>::Symbolic as KnownType>::TY,
                                arg2: <<$t2 as KnownType>::Symbolic as KnownType>::TY,
                                ret: <<$u as KnownType>::Symbolic as KnownType>::TY
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into().unwrap();
                            let op = self.clone();

                            let k = <$op as TernaryKernel<
                                SymbolicContext,
                                $plc,
                                <$t0 as KnownType>::Symbolic,
                                <$t1 as KnownType>::Symbolic,
                                <$t2 as KnownType>::Symbolic,
                                <$u as KnownType>::Symbolic,
                            >>::compile(self, &ctx, &plc);

                            Box::new(move |operands| {
                                assert_eq!(operands.len(), 3);

                                let x0: <$t0 as KnownType>::Symbolic = operands.get(0).unwrap().clone().try_into().unwrap();
                                let x1: <$t1 as KnownType>::Symbolic = operands.get(1).unwrap().clone().try_into().unwrap();
                                let x2: <$t2 as KnownType>::Symbolic = operands.get(2).unwrap().clone().try_into().unwrap();

                                let y: <$u as KnownType>::Symbolic = k(&ctx, &plc, x0, x1, x2);
                                SymbolicValue::from(y)
                            })
                        }
                    )+
                    _ => unimplemented!(), // ok
                }
            }

        }
    };
}


/// Kernel function maybe be evaluated in symbolic contexts
macro_rules! hybrid_kernel {

    /*
    Nullary
    */

    ($op:ty, [$( ($plc:ty, () -> $u:ty => $($kp:tt)+), )+]) => {
        concrete_dispatch_kernel!($op, [$( ($plc, () -> $u), )+]);
        symbolic_dispatch_kernel!($op, [$( ($plc, () -> $u), )+]);

        $(
            impl NullaryKernel<
                ConcreteContext,
                $plc,
                $u
            > for $op
            {
                fn compile(&self, ctx: &ConcreteContext, plc: &$plc) -> Box<dyn Fn(
                    &ConcreteContext,
                    &$plc)
                    -> $u>
                {
                    derive_runtime_kernel![nullary, $($kp)+, self]
                }
            }
        )+

        $(
            impl NullaryKernel<
                SymbolicContext,
                $plc,
                <$u as KnownType>::Symbolic
            > for $op
            {
                fn compile(&self, ctx: &SymbolicContext, plc: &$plc) -> Box<dyn Fn(
                    &SymbolicContext,
                    &$plc)
                    -> <$u as KnownType>::Symbolic>
                {
                    let k = derive_runtime_kernel![nullary, $($kp)+, self];

                    Box::new(move |
                        ctx: &SymbolicContext,
                        plc: &$plc,
                    | {
                        let y = k(ctx, &plc);
                        y.into()
                    })
                }
            }
        )+
    };

    /*
    Unary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty) -> $u:ty => $($kp:tt)+), )+]) => {
        concrete_dispatch_kernel!($op, [$( ($plc, ($t0) -> $u), )+]);
        symbolic_dispatch_kernel!($op, [$( ($plc, ($t0) -> $u), )+]);

        $(
            impl UnaryKernel<
                ConcreteContext,
                $plc,
                $t0,
                $u
            > for $op
            {
                fn compile(&self, ctx: &ConcreteContext, plc: &$plc) -> Box<dyn Fn(&ConcreteContext, &$plc, $t0) -> $u> {
                    derive_runtime_kernel![unary, $($kp)+, self]
                }
            }
        )+

        $(
            impl UnaryKernel<
                SymbolicContext,
                $plc,
                <$t0 as KnownType>::Symbolic,
                <$u as KnownType>::Symbolic
            > for $op
            {
                fn compile(&self, ctx: &SymbolicContext, plc: &$plc) -> Box<dyn Fn(
                    &SymbolicContext,
                    &$plc,
                    <$t0 as KnownType>::Symbolic)
                    -> <$u as KnownType>::Symbolic>
                {
                    let k = derive_runtime_kernel![unary, $($kp)+, self];

                    let op = self.clone();
                    Box::new(move |
                        ctx: &SymbolicContext,
                        plc: &$plc,
                        x0: <$t0 as KnownType>::Symbolic,
                    | {
                        let v0 = x0.clone().try_into();

                        match v0 {
                            Ok(v0) => {
                                let y = k(ctx, &plc, v0);
                                y.into()
                            }
                            _ => match x0 {
                                Symbolic::Symbolic(h0) => {
                                    let op_name = ctx.add_operation(&op, &[&h0.op], &plc.clone().into());
                                    Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.clone().into() })
                                }
                                _ => unimplemented!() // ok
                            }
                        }
                    })
                }
            }
        )+
    };

    /*
    Binary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty, $t1:ty) -> $u:ty => $($kp:tt)+), )+]) => {
        concrete_dispatch_kernel!($op, [$( ($plc, ($t0, $t1) -> $u), )+]);
        symbolic_dispatch_kernel!($op, [$( ($plc, ($t0, $t1) -> $u), )+]);

        $(
            impl BinaryKernel<
                ConcreteContext,
                $plc,
                $t0,
                $t1,
                $u
            > for $op
            {
                fn compile(&self, ctx: &ConcreteContext, plc: &$plc) -> Box<dyn Fn(&ConcreteContext, &$plc, $t0, $t1) -> $u> {
                    derive_runtime_kernel![binary, $($kp)+, self]
                }
            }
        )+

        $(
            impl BinaryKernel<
                SymbolicContext,
                $plc,
                <$t0 as KnownType>::Symbolic,
                <$t1 as KnownType>::Symbolic,
                <$u as KnownType>::Symbolic
            > for $op
            {
                fn compile(&self, ctx: &SymbolicContext, plc: &$plc) -> Box<dyn Fn(
                    &SymbolicContext,
                    &$plc,
                    <$t0 as KnownType>::Symbolic,
                    <$t1 as KnownType>::Symbolic)
                    -> <$u as KnownType>::Symbolic>
                {
                    let k = derive_runtime_kernel![binary, $($kp)+, self];

                    let op = self.clone();
                    Box::new(move |
                        ctx: &SymbolicContext,
                        plc: &$plc,
                        x0: <$t0 as KnownType>::Symbolic,
                        x1: <$t1 as KnownType>::Symbolic,
                    | {
                        let v0 = x0.clone().try_into();
                        let v1 = x1.clone().try_into();

                        match (v0, v1) {
                            (Ok(v0), Ok(v1)) => {
                                let y = k(ctx, &plc, v0, v1);
                                y.into()
                            }
                            _ => match (x0, x1) {
                                (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1)) => {
                                    let op_name = ctx.add_operation(&op, &[&h0.op, &h1.op], &plc.clone().into());
                                    Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.clone().into() })
                                }
                                _ => unimplemented!() // ok
                            }
                        }
                    })
                }
            }
        )+
    };

    /*
    Ternary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => $($kp:tt)+), )+]) => {
        concrete_dispatch_kernel!($op, [$( ($plc, ($t0, $t1, $t2) -> $u), )+]);
        symbolic_dispatch_kernel!($op, [$( ($plc, ($t0, $t1, $t2) -> $u), )+]);

        $(
            impl TernaryKernel<
                ConcreteContext,
                $plc,
                $t0,
                $t1,
                $t2,
                $u
            > for $op
            {
                fn compile(&self, ctx: &ConcreteContext, plc: &$plc) -> Box<dyn Fn(&ConcreteContext, &$plc, $t0, $t1, $t2) -> $u> {
                    derive_runtime_kernel![ternary, $($kp)+, self]
                }
            }
        )+

        $(
            impl TernaryKernel<
                SymbolicContext,
                $plc,
                <$t0 as KnownType>::Symbolic,
                <$t1 as KnownType>::Symbolic,
                <$t2 as KnownType>::Symbolic,
                <$u as KnownType>::Symbolic
            > for $op
            {
                fn compile(&self, ctx: &SymbolicContext, plc: &$plc) -> Box<dyn Fn(
                    &SymbolicContext,
                    &$plc,
                    <$t0 as KnownType>::Symbolic,
                    <$t1 as KnownType>::Symbolic,
                    <$t2 as KnownType>::Symbolic)
                    -> <$u as KnownType>::Symbolic>
                {
                    let k = derive_runtime_kernel![ternary, $($kp)+, self];

                    let op = self.clone();
                    Box::new(move |
                        ctx: &SymbolicContext,
                        plc: &$plc,
                        x0: <$t0 as KnownType>::Symbolic,
                        x1: <$t1 as KnownType>::Symbolic,
                        x2: <$t2 as KnownType>::Symbolic,
                    | {
                        let v0 = x0.clone().try_into();
                        let v1 = x1.clone().try_into();
                        let v2 = x2.clone().try_into();

                        match (v0, v1, v2) {
                            (Ok(v0), Ok(v1), Ok(v2)) => {
                                let y = k(ctx, &plc, v0, v1, v2);
                                y.into()
                            }
                            _ => match (x0, x1, x2) {
                                (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1), Symbolic::Symbolic(h2)) => {
                                    let op_name = ctx.add_operation(&op, &[&h0.op, &h1.op, &h2.op], &plc.clone().into());
                                    Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.clone().into() })
                                }
                                _ => unimplemented!() // ok
                            }
                        }
                    })
                }
            }
        )+
    };
}


trait NullaryKernelCheck<C: Context, P, Y>
where
    Self: NullaryKernel<C, P, Y>,
{
}

trait UnaryKernelCheck<C: Context, P, X0, Y>
where
    Self: UnaryKernel<C, P, X0, Y>,
{
}

trait BinaryKernelCheck<C: Context, P, X0, X1, Y>
where
    Self: BinaryKernel<C, P, X0, X1, Y>,
{
}

trait TernaryKernelCheck<C: Context, P, X0, X1, X2, Y>
where
    Self: TernaryKernel<C, P, X0, X1, X2, Y>,
{
}

hybrid_kernel! {
    RepSetupOp,
    [
        (ReplicatedPlacement, () -> ReplicatedSetup => Self::kernel),
    ]
}

modelled!(PlacementRepToAdd::rep_to_add, AdditivePlacement, (Replicated64Tensor) -> Additive64Tensor, RepToAddOp);
modelled!(PlacementRepToAdd::rep_to_add, AdditivePlacement, (Replicated128Tensor) -> Additive128Tensor, RepToAddOp);

hybrid_kernel! {
    RepToAddOp,
    [
        (AdditivePlacement, (Replicated64Tensor) -> Additive64Tensor => Self::rep_to_add_kernel),
        (AdditivePlacement, (Replicated128Tensor) -> Additive128Tensor => Self::rep_to_add_kernel),
    ]
}

modelled!(PlacementAdd::add, ReplicatedPlacement, (Replicated64Tensor, Replicated64Tensor) -> Replicated64Tensor, RepAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (Replicated128Tensor, Replicated128Tensor) -> Replicated128Tensor, RepAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (Ring64Tensor, Replicated64Tensor) -> Replicated64Tensor, RepAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (Ring128Tensor, Replicated128Tensor) -> Replicated128Tensor, RepAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (Replicated64Tensor, Ring64Tensor) -> Replicated64Tensor, RepAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (Replicated128Tensor, Ring128Tensor) -> Replicated128Tensor, RepAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor, RepAddOp);

hybrid_kernel! {
    RepAddOp,
    [
        (ReplicatedPlacement, (Replicated64Tensor, Replicated64Tensor) -> Replicated64Tensor => Self::rep_rep_kernel),
        (ReplicatedPlacement, (Replicated128Tensor, Replicated128Tensor) -> Replicated128Tensor => Self::rep_rep_kernel),
        (ReplicatedPlacement, (Ring64Tensor, Replicated64Tensor) -> Replicated64Tensor => Self::ring_rep_kernel),
        (ReplicatedPlacement, (Ring128Tensor, Replicated128Tensor) -> Replicated128Tensor => Self::ring_rep_kernel),
        (ReplicatedPlacement, (Replicated64Tensor, Ring64Tensor) -> Replicated64Tensor => Self::rep_ring_kernel),
        (ReplicatedPlacement, (Replicated128Tensor, Ring128Tensor) -> Replicated128Tensor => Self::rep_ring_kernel),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor => Self::rep_rep_kernel),
    ]
}

modelled!(PlacementAdd::add, AdditivePlacement, (Additive64Tensor, Additive64Tensor) -> Additive64Tensor, AdditiveAddOp);
modelled!(PlacementAdd::add, AdditivePlacement, (Additive128Tensor, Additive128Tensor) -> Additive128Tensor, AdditiveAddOp);

hybrid_kernel! {
    AdditiveAddOp,
    [
        (AdditivePlacement, (Additive64Tensor, Additive64Tensor) -> Additive64Tensor => Self::add_add_kernel),
        (AdditivePlacement, (Additive128Tensor, Additive128Tensor) -> Additive128Tensor => Self::add_add_kernel),
        (AdditivePlacement, (Additive64Tensor, Ring64Tensor) -> Additive64Tensor => Self::add_ring_kernel),
        (AdditivePlacement, (Additive128Tensor, Ring128Tensor) -> Additive128Tensor => Self::add_ring_kernel),
        (AdditivePlacement, (Ring64Tensor, Additive64Tensor) -> Additive64Tensor => Self::ring_add_kernel),
        (AdditivePlacement, (Ring128Tensor, Additive128Tensor) -> Additive128Tensor => Self::ring_add_kernel),
    ]
}

modelled!(PlacementMulSetup::mul, ReplicatedPlacement, (ReplicatedSetup, Replicated64Tensor, Replicated64Tensor) -> Replicated64Tensor, RepMulOp);
modelled!(PlacementMulSetup::mul, ReplicatedPlacement, (ReplicatedSetup, Replicated128Tensor, Replicated128Tensor) -> Replicated128Tensor, RepMulOp);
modelled!(PlacementMulSetup::mul, ReplicatedPlacement, (ReplicatedSetup, Ring64Tensor, Replicated64Tensor) -> Replicated64Tensor, RepMulOp);
modelled!(PlacementMulSetup::mul, ReplicatedPlacement, (ReplicatedSetup, Ring128Tensor, Replicated128Tensor) -> Replicated128Tensor, RepMulOp);
modelled!(PlacementMulSetup::mul, ReplicatedPlacement, (ReplicatedSetup, Replicated64Tensor, Ring64Tensor) -> Replicated64Tensor, RepMulOp);
modelled!(PlacementMulSetup::mul, ReplicatedPlacement, (ReplicatedSetup, Replicated128Tensor, Ring128Tensor) -> Replicated128Tensor, RepMulOp);
modelled!(PlacementMulSetup::mul, ReplicatedPlacement, (ReplicatedSetup, ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor, RepMulOp);

hybrid_kernel! {
    RepMulOp,
    [
        (ReplicatedPlacement, (ReplicatedSetup, Replicated64Tensor, Replicated64Tensor) -> Replicated64Tensor => Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, Replicated128Tensor, Replicated128Tensor) -> Replicated128Tensor => Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor => Self::rep_rep_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, Ring64Tensor, Replicated64Tensor) -> Replicated64Tensor => Self::ring_rep_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, Ring128Tensor, Replicated128Tensor) -> Replicated128Tensor => Self::ring_rep_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, Replicated64Tensor, Ring64Tensor) -> Replicated64Tensor => Self::rep_ring_kernel),
        (ReplicatedPlacement, (ReplicatedSetup, Replicated128Tensor, Ring128Tensor) -> Replicated128Tensor => Self::rep_ring_kernel),
    ]
}

modelled!(PlacementMul::mul, AdditivePlacement, (Ring64Tensor, Additive64Tensor) -> Additive64Tensor, AdditiveMulOp);
modelled!(PlacementMul::mul, AdditivePlacement, (Additive64Tensor, Ring64Tensor) -> Additive64Tensor, AdditiveMulOp);
modelled!(PlacementMul::mul, AdditivePlacement, (Ring128Tensor, Additive128Tensor) -> Additive128Tensor, AdditiveMulOp);
modelled!(PlacementMul::mul, AdditivePlacement, (Additive128Tensor, Ring128Tensor) -> Additive128Tensor, AdditiveMulOp);

hybrid_kernel! {
    AdditiveMulOp,
    [
        (AdditivePlacement, (Ring64Tensor, Additive64Tensor) -> Additive64Tensor => Self::ring_add_kernel),
        (AdditivePlacement, (Additive64Tensor, Ring64Tensor) -> Additive64Tensor => Self::add_ring_kernel),
        (AdditivePlacement, (Additive128Tensor, Ring128Tensor) -> Additive128Tensor => Self::add_ring_kernel),
        (AdditivePlacement, (Ring128Tensor, Additive128Tensor) -> Additive128Tensor => Self::ring_add_kernel),
    ]
}

modelled!(PlacementShare::share, ReplicatedPlacement, (Ring64Tensor) -> Replicated64Tensor, RepShareOp);
modelled!(PlacementShare::share, ReplicatedPlacement, (Ring128Tensor) -> Replicated128Tensor, RepShareOp);
modelled!(PlacementShare::share, ReplicatedPlacement, (BitTensor) -> ReplicatedBitTensor, RepShareOp);

hybrid_kernel! {
    RepShareOp,
    [
        (ReplicatedPlacement, (Ring64Tensor) -> Replicated64Tensor => Self::kernel),
        (ReplicatedPlacement, (Ring128Tensor) -> Replicated128Tensor => Self::kernel),
        (ReplicatedPlacement, (BitTensor) -> ReplicatedBitTensor => Self::kernel),
    ]
}

// NOTE
// revealing on ReplicatedPlacements should reveal to all three players, but we're currently
// missing a type to represent this (eg PublicReplicatedTensor vs PrivateReplicatedTensors)
modelled!(PlacementReveal::reveal, HostPlacement, (Replicated64Tensor) -> Ring64Tensor, RepRevealOp);
modelled!(PlacementReveal::reveal, HostPlacement, (Replicated128Tensor) -> Ring128Tensor, RepRevealOp);
modelled!(PlacementReveal::reveal, HostPlacement, (ReplicatedBitTensor) -> BitTensor, RepRevealOp);

hybrid_kernel! {
    RepRevealOp,
    [
        (HostPlacement, (Replicated64Tensor) -> Ring64Tensor => Self::kernel),
        (HostPlacement, (Replicated128Tensor) -> Ring128Tensor => Self::kernel),
        (HostPlacement, (ReplicatedBitTensor) -> BitTensor => Self::kernel),
    ]
}

modelled!(PlacementReveal::reveal, HostPlacement, (Additive64Tensor) -> Ring64Tensor, AdditiveRevealOp);
modelled!(PlacementReveal::reveal, HostPlacement, (Additive128Tensor) -> Ring128Tensor, AdditiveRevealOp);

hybrid_kernel! {
    AdditiveRevealOp,
    [
        (HostPlacement, (Additive64Tensor) -> Ring64Tensor => Self::kernel),
        (HostPlacement, (Additive128Tensor) -> Ring128Tensor => Self::kernel),
    ]
}

modelled!(PlacementAdd::add, HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor, RingAddOp);
modelled!(PlacementAdd::add, HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor, RingAddOp);

modelled!(PlacementSub::sub, HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor, RingSubOp);
modelled!(PlacementSub::sub, HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor, RingSubOp);

modelled!(PlacementMul::mul, HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor, RingMulOp);
modelled!(PlacementMul::mul, HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor, RingMulOp);

modelled!(PlacementShl::shl, HostPlacement, attributes[amount: usize] (Ring64Tensor) -> Ring64Tensor, RingShlOp);
modelled!(PlacementShl::shl, HostPlacement, attributes[amount: usize] (Ring128Tensor) -> Ring128Tensor, RingShlOp);

modelled!(PlacementShr::shr, HostPlacement, attributes[amount: usize] (Ring64Tensor) -> Ring64Tensor, RingShrOp);
modelled!(PlacementShr::shr, HostPlacement, attributes[amount: usize] (Ring128Tensor) -> Ring128Tensor, RingShrOp);

modelled!(PlacementXor::xor, HostPlacement, (BitTensor, BitTensor) -> BitTensor, BitXorOp);
modelled_alias!(PlacementAdd::add, HostPlacement, (BitTensor, BitTensor) -> BitTensor => PlacementXor::xor); // add = xor in Z2
modelled_alias!(PlacementSub::sub, HostPlacement, (BitTensor, BitTensor) -> BitTensor => PlacementXor::xor); // sub = xor in Z2

modelled!(PlacementAnd::and, HostPlacement, (BitTensor, BitTensor) -> BitTensor, BitAndOp);
modelled_alias!(PlacementMul::mul, HostPlacement, (BitTensor, BitTensor) -> BitTensor => PlacementAnd::and); // mul = and in Z2

modelled!(PlacementKeyGen::keygen, HostPlacement, () -> PrfKey, PrfKeyGenOp);

modelled!(PlacementSample::sample, HostPlacement, () -> Ring64Tensor, RingSampleOp);
modelled!(PlacementSample::sample, HostPlacement, () -> Ring128Tensor, RingSampleOp);

modelled!(PlacementSample::sample, HostPlacement, () -> BitTensor, BitSampleOp);

impl DispatchKernel<ConcreteContext> for ConstantOp {
    fn compile(&self, _ctx: &ConcreteContext, plc: &Placement) -> Box<dyn Fn(Vec<Value>) -> Value> {
        let val = self.val.clone();

        match plc {
            Placement::HostPlacement(_) => Box::new(move |_operands| -> Value { *val.clone() }),
            _ => unimplemented!(), // ok
        }
    }
}

impl DispatchKernel<SymbolicContext> for ConstantOp {
    fn compile<'c>(
        &self,
        ctx: &'c SymbolicContext,
        plc: &Placement,
    ) -> Box<dyn Fn(Vec<SymbolicValue>) -> SymbolicValue + 'c> {
        match plc {
            Placement::HostPlacement(_) => {
                // TODO
                let plc = plc.clone();
                let op = self.clone();

                Box::new(move |operands| {
                    assert_eq!(operands.len(), 0);

                    let op_name = ctx.add_operation(&op, &[], &plc);
                    op.val.ty().synthesize_symbolic_value(op_name, plc.clone())
                })
            }
            _ => unimplemented!(), // ok
        }
    }
}

modelled!(PlacementMul::mul, HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedMulOp);
modelled!(PlacementMul::mul, HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedMulOp);
modelled!(PlacementMul::mul, ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedMulOp);
modelled!(PlacementMul::mul, ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedMulOp);

hybrid_kernel! {
    FixedMulOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => Self::host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => Self::host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => Self::rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => Self::rep_kernel),
    ]
}

impl FixedMulOp {
    fn host_kernel<C: Context, RingTensorT, ReplicatedTensorT>(
        ctx: &C,
        plc: &HostPlacement,
        x: FixedTensor<RingTensorT, ReplicatedTensorT>,
        y: FixedTensor<RingTensorT, ReplicatedTensorT>,
    ) -> FixedTensor<RingTensorT, ReplicatedTensorT>
    where
        HostPlacement: PlacementReveal<C, ReplicatedTensorT, Output = RingTensorT>,
        HostPlacement: PlacementMul<C, RingTensorT, RingTensorT, Output = RingTensorT>,
    {
        // NOTE: if one day we have branches that are not supported then we should
        // consider promoting matching to the macros and introduce proper intermediate types

        match (x, y) {
            (FixedTensor::RingTensor(x), FixedTensor::RingTensor(y)) => {
                let z: RingTensorT = plc.mul(ctx, &x, &y);
                FixedTensor::<RingTensorT, ReplicatedTensorT>::RingTensor(z)
            }
            (FixedTensor::RingTensor(x), FixedTensor::ReplicatedTensor(ye)) => {
                let y = plc.reveal(ctx, &ye);
                let z = plc.mul(ctx, &x, &y);
                FixedTensor::<RingTensorT, ReplicatedTensorT>::RingTensor(z)
            }
            (FixedTensor::ReplicatedTensor(xe), FixedTensor::RingTensor(y)) => {
                let x = plc.reveal(ctx, &xe);
                let z = plc.mul(ctx, &x, &y);
                FixedTensor::<RingTensorT, ReplicatedTensorT>::RingTensor(z)
            }
            (FixedTensor::ReplicatedTensor(xe), FixedTensor::ReplicatedTensor(ye)) => {
                let x = plc.reveal(ctx, &xe);
                let y = plc.reveal(ctx, &ye);
                let z = plc.mul(ctx, &x, &y);
                FixedTensor::<RingTensorT, ReplicatedTensorT>::RingTensor(z)
            }
        }
    }

    fn rep_kernel<C: Context, RingTensorT, ReplicatedTensorT>(
        ctx: &C,
        plc: &ReplicatedPlacement,
        x: FixedTensor<RingTensorT, ReplicatedTensorT>,
        y: FixedTensor<RingTensorT, ReplicatedTensorT>,
    ) -> FixedTensor<RingTensorT, ReplicatedTensorT>
    where
        ReplicatedPlacement: PlacementShare<C, RingTensorT, Output = ReplicatedTensorT>,
        ReplicatedPlacement: PlacementMulSetup<
            C,
            C::ReplicatedSetup,
            ReplicatedTensorT,
            ReplicatedTensorT,
            Output = ReplicatedTensorT,
        >,
        ReplicatedPlacement:
            PlacementAdd<C, ReplicatedTensorT, ReplicatedTensorT, Output = ReplicatedTensorT>,
    {
        // NOTE: if one day we have branches that are not supported then we should
        // consider promoting matching to the macros and introduce proper intermediate types

        match (x, y) {
            (FixedTensor::RingTensor(x), FixedTensor::RingTensor(y)) => {
                let setup = ctx.replicated_setup(plc);
                let xe = plc.share(ctx, &x);
                let ye = plc.share(ctx, &y);
                let ze = PlacementMulSetup::mul(plc, ctx, setup, &xe, &ye);
                FixedTensor::<RingTensorT, ReplicatedTensorT>::ReplicatedTensor(ze)
            }
            (FixedTensor::RingTensor(x), FixedTensor::ReplicatedTensor(ye)) => {
                let setup = ctx.replicated_setup(plc);
                let xe = plc.share(ctx, &x);
                let ze = PlacementMulSetup::mul(plc, ctx, setup, &xe, &ye);
                FixedTensor::<RingTensorT, ReplicatedTensorT>::ReplicatedTensor(ze)
            }
            (FixedTensor::ReplicatedTensor(xe), FixedTensor::RingTensor(y)) => {
                let setup = ctx.replicated_setup(plc);
                let ye = plc.share(ctx, &y);
                let ze = PlacementMulSetup::mul(plc, ctx, setup, &xe, &ye);
                FixedTensor::<RingTensorT, ReplicatedTensorT>::ReplicatedTensor(ze)
            }
            (FixedTensor::ReplicatedTensor(xe), FixedTensor::ReplicatedTensor(ye)) => {
                let setup = ctx.replicated_setup(plc);
                let ze = PlacementMulSetup::mul(plc, ctx, setup, &xe, &ye);
                FixedTensor::<RingTensorT, ReplicatedTensorT>::ReplicatedTensor(ze)
            }
        }
    }
}

modelled!(PlacementAdd::add, HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedAddOp);
modelled!(PlacementAdd::add, HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor, FixedAddOp);
modelled!(PlacementAdd::add, ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor, FixedAddOp);

hybrid_kernel! {
    FixedAddOp,
    [
        (HostPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => Self::host_kernel),
        (HostPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => Self::host_kernel),
        (ReplicatedPlacement, (Fixed64Tensor, Fixed64Tensor) -> Fixed64Tensor => Self::rep_kernel),
        (ReplicatedPlacement, (Fixed128Tensor, Fixed128Tensor) -> Fixed128Tensor => Self::rep_kernel),
    ]
}

impl FixedAddOp {
    fn host_kernel<C: Context, RingTensorT, ReplicatedTensorT>(
        ctx: &C,
        plc: &HostPlacement,
        x: FixedTensor<RingTensorT, ReplicatedTensorT>,
        y: FixedTensor<RingTensorT, ReplicatedTensorT>,
    ) -> FixedTensor<RingTensorT, ReplicatedTensorT>
    where
        HostPlacement: PlacementReveal<C, ReplicatedTensorT, Output = RingTensorT>,
        HostPlacement: PlacementAdd<C, RingTensorT, RingTensorT, Output = RingTensorT>,
    {
        // NOTE: if one day we have branches that are not supported then we should
        // consider promoting matching to the macros and introduce proper intermediate types

        match (x, y) {
            (FixedTensor::RingTensor(x), FixedTensor::RingTensor(y)) => {
                let z: RingTensorT = plc.add(ctx, &x, &y);
                FixedTensor::<RingTensorT, ReplicatedTensorT>::RingTensor(z)
            }
            (FixedTensor::RingTensor(x), FixedTensor::ReplicatedTensor(ye)) => {
                let y = plc.reveal(ctx, &ye);
                let z = plc.add(ctx, &x, &y);
                FixedTensor::<RingTensorT, ReplicatedTensorT>::RingTensor(z)
            }
            (FixedTensor::ReplicatedTensor(xe), FixedTensor::RingTensor(y)) => {
                let x = plc.reveal(ctx, &xe);
                let z = plc.add(ctx, &x, &y);
                FixedTensor::<RingTensorT, ReplicatedTensorT>::RingTensor(z)
            }
            (FixedTensor::ReplicatedTensor(xe), FixedTensor::ReplicatedTensor(ye)) => {
                let x = plc.reveal(ctx, &xe);
                let y = plc.reveal(ctx, &ye);
                let z = plc.add(ctx, &x, &y);
                FixedTensor::<RingTensorT, ReplicatedTensorT>::RingTensor(z)
            }
        }
    }

    fn rep_kernel<C: Context, RingTensorT, ReplicatedTensorT>(
        ctx: &C,
        plc: &ReplicatedPlacement,
        x: FixedTensor<RingTensorT, ReplicatedTensorT>,
        y: FixedTensor<RingTensorT, ReplicatedTensorT>,
    ) -> FixedTensor<RingTensorT, ReplicatedTensorT>
    where
        ReplicatedPlacement: PlacementShare<C, RingTensorT, Output = ReplicatedTensorT>,
        ReplicatedPlacement:
            PlacementAdd<C, ReplicatedTensorT, ReplicatedTensorT, Output = ReplicatedTensorT>,
    {
        // NOTE: if one day we have branches that are not supported then we should
        // consider promoting matching to the macros and introduce proper intermediate types

        match (x, y) {
            (FixedTensor::RingTensor(x), FixedTensor::RingTensor(y)) => {
                let xe = plc.share(ctx, &x);
                let ye = plc.share(ctx, &y);
                let ze = plc.add(ctx, &xe, &ye);
                FixedTensor::<RingTensorT, ReplicatedTensorT>::ReplicatedTensor(ze)
            }
            (FixedTensor::RingTensor(x), FixedTensor::ReplicatedTensor(ye)) => {
                let xe = plc.share(ctx, &x);
                let ze = plc.add(ctx, &xe, &ye);
                FixedTensor::<RingTensorT, ReplicatedTensorT>::ReplicatedTensor(ze)
            }
            (FixedTensor::ReplicatedTensor(xe), FixedTensor::RingTensor(y)) => {
                let ye = plc.share(ctx, &y);
                let ze = plc.add(ctx, &xe, &ye);
                FixedTensor::<RingTensorT, ReplicatedTensorT>::ReplicatedTensor(ze)
            }
            (FixedTensor::ReplicatedTensor(xe), FixedTensor::ReplicatedTensor(ye)) => {
                let ze = plc.add(ctx, &xe, &ye);
                FixedTensor::<RingTensorT, ReplicatedTensorT>::ReplicatedTensor(ze)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::redundant_clone)]

    use super::*;

    #[test]
    fn test_rep_add_concrete() {
        let ctx = ConcreteContext::default();

        let alice = HostPlacement {
            player: "alice".into(),
        };
        let bob = HostPlacement {
            player: "bob".into(),
        };
        let carole = HostPlacement {
            player: "carole".into(),
        };
        let rep = ReplicatedPlacement {
            players: ["alice".into(), "bob".into(), "carole".into()],
        };

        let xe: Replicated64Tensor = ReplicatedTensor {
            shares: [
                [RingTensor(1, alice.clone()), RingTensor(2, alice.clone())],
                [RingTensor(2, bob.clone()), RingTensor(3, bob.clone())],
                [RingTensor(3, carole.clone()), RingTensor(1, carole.clone())],
            ],
        };

        let ye = ReplicatedTensor {
            shares: [
                [RingTensor(1, alice.clone()), RingTensor(2, alice.clone())],
                [RingTensor(2, bob.clone()), RingTensor(3, bob.clone())],
                [RingTensor(3, carole.clone()), RingTensor(1, carole.clone())],
            ],
        };

        let ze: ReplicatedTensor<_> = rep.add(&ctx, &xe, &ye);

        assert_eq!(
            ze,
            ReplicatedTensor {
                shares: [
                    [RingTensor(2, alice.clone()), RingTensor(4, alice.clone())],
                    [RingTensor(4, bob.clone()), RingTensor(6, bob.clone())],
                    [RingTensor(6, carole.clone()), RingTensor(2, carole.clone())],
                ],
            }
        );
    }

    #[test]
    fn test_rep_add_symbolic() {
        let ctx = SymbolicContext::default();

        let alice = HostPlacement {
            player: "alice".into(),
        };
        let bob = HostPlacement {
            player: "bob".into(),
        };
        let carole = HostPlacement {
            player: "carole".into(),
        };
        let rep = ReplicatedPlacement {
            players: ["alice".into(), "bob".into(), "carole".into()],
        };

        let xe: Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>> =
            Symbolic::Concrete(ReplicatedTensor {
                shares: [
                    [
                        SymbolicHandle {
                            op: "x00".into(),
                            plc: alice.clone(),
                        }
                        .into(),
                        SymbolicHandle {
                            op: "x10".into(),
                            plc: alice.clone(),
                        }
                        .into(),
                    ],
                    [
                        SymbolicHandle {
                            op: "x11".into(),
                            plc: bob.clone(),
                        }
                        .into(),
                        SymbolicHandle {
                            op: "x21".into(),
                            plc: bob.clone(),
                        }
                        .into(),
                    ],
                    [
                        SymbolicHandle {
                            op: "x22".into(),
                            plc: carole.clone(),
                        }
                        .into(),
                        SymbolicHandle {
                            op: "x02".into(),
                            plc: carole.clone(),
                        }
                        .into(),
                    ],
                ],
            });

        let ye: Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>> =
            Symbolic::Concrete(ReplicatedTensor {
                shares: [
                    [
                        SymbolicHandle {
                            op: "y00".into(),
                            plc: alice.clone(),
                        }
                        .into(),
                        SymbolicHandle {
                            op: "y10".into(),
                            plc: alice.clone(),
                        }
                        .into(),
                    ],
                    [
                        SymbolicHandle {
                            op: "y11".into(),
                            plc: bob.clone(),
                        }
                        .into(),
                        SymbolicHandle {
                            op: "y21".into(),
                            plc: bob.clone(),
                        }
                        .into(),
                    ],
                    [
                        SymbolicHandle {
                            op: "y22".into(),
                            plc: carole.clone(),
                        }
                        .into(),
                        SymbolicHandle {
                            op: "y02".into(),
                            plc: carole.clone(),
                        }
                        .into(),
                    ],
                ],
            });

        let ze = rep.add(&ctx, &xe, &ye);

        assert_eq!(
            ze,
            Symbolic::Concrete(ReplicatedTensor {
                shares: [
                    [
                        Symbolic::Symbolic(SymbolicHandle {
                            op: "op_0".into(),
                            plc: alice.clone()
                        }),
                        Symbolic::Symbolic(SymbolicHandle {
                            op: "op_1".into(),
                            plc: alice.clone()
                        }),
                    ],
                    [
                        Symbolic::Symbolic(SymbolicHandle {
                            op: "op_2".into(),
                            plc: bob.clone()
                        }),
                        Symbolic::Symbolic(SymbolicHandle {
                            op: "op_3".into(),
                            plc: bob.clone()
                        }),
                    ],
                    [
                        Symbolic::Symbolic(SymbolicHandle {
                            op: "op_4".into(),
                            plc: carole.clone()
                        }),
                        Symbolic::Symbolic(SymbolicHandle {
                            op: "op_5".into(),
                            plc: carole.clone()
                        }),
                    ],
                ]
            })
        );

        let ops: &[_] = &ctx.ops.read().unwrap();
        assert_eq!(
            ops,
            &vec![
                Operation {
                    name: "op_0".into(),
                    operator: RingAddOp {
                        sig: BinarySignature {
                            arg0: Ty::Ring64Tensor,
                            arg1: Ty::Ring64Tensor,
                            ret: Ty::Ring64Tensor
                        }
                        .into(),
                    }
                    .into(),
                    operands: vec!["x00".into(), "y00".into()],
                    plc: HostPlacement {
                        player: "alice".into()
                    }
                    .into(),
                },
                Operation {
                    name: "op_1".into(),
                    operator: RingAddOp {
                        sig: BinarySignature {
                            arg0: Ty::Ring64Tensor,
                            arg1: Ty::Ring64Tensor,
                            ret: Ty::Ring64Tensor
                        }
                        .into(),
                    }
                    .into(),
                    operands: vec!["x10".into(), "y10".into()],
                    plc: HostPlacement {
                        player: "alice".into()
                    }
                    .into(),
                },
                Operation {
                    name: "op_2".into(),
                    operator: RingAddOp {
                        sig: BinarySignature {
                            arg0: Ty::Ring64Tensor,
                            arg1: Ty::Ring64Tensor,
                            ret: Ty::Ring64Tensor
                        }
                        .into(),
                    }
                    .into(),
                    operands: vec!["x11".into(), "y11".into()],
                    plc: HostPlacement {
                        player: "bob".into()
                    }
                    .into(),
                },
                Operation {
                    name: "op_3".into(),
                    operator: RingAddOp {
                        sig: BinarySignature {
                            arg0: Ty::Ring64Tensor,
                            arg1: Ty::Ring64Tensor,
                            ret: Ty::Ring64Tensor
                        }
                        .into(),
                    }
                    .into(),
                    operands: vec!["x21".into(), "y21".into()],
                    plc: HostPlacement {
                        player: "bob".into()
                    }
                    .into(),
                },
                Operation {
                    name: "op_4".into(),
                    operator: RingAddOp {
                        sig: BinarySignature {
                            arg0: Ty::Ring64Tensor,
                            arg1: Ty::Ring64Tensor,
                            ret: Ty::Ring64Tensor
                        }
                        .into(),
                    }
                    .into(),
                    operands: vec!["x22".into(), "y22".into()],
                    plc: HostPlacement {
                        player: "carole".into()
                    }
                    .into(),
                },
                Operation {
                    name: "op_5".into(),
                    operator: RingAddOp {
                        sig: BinarySignature {
                            arg0: Ty::Ring64Tensor,
                            arg1: Ty::Ring64Tensor,
                            ret: Ty::Ring64Tensor
                        }
                        .into(),
                    }
                    .into(),
                    operands: vec!["x02".into(), "y02".into()],
                    plc: HostPlacement {
                        player: "carole".into()
                    }
                    .into(),
                },
            ]
        );
    }

    #[test]
    fn test_rep_share_concrete() {
        let alice = HostPlacement {
            player: "alice".into(),
        };
        let bob = HostPlacement {
            player: "bob".into(),
        };
        let carole = HostPlacement {
            player: "carole".into(),
        };
        let rep = ReplicatedPlacement {
            players: ["alice".into(), "bob".into(), "carole".into()],
        };

        let replicated_keys = HashMap::new();
        let ctx = ConcreteContext { replicated_keys };

        let x: Ring64Tensor = RingTensor(5, alice);
        let xe = rep.share(&ctx, &x);
    }

    #[test]
    fn test_rep_share_symbolic() {
        let alice_plc = HostPlacement {
            player: "alice".into(),
        };
        let bob_plc = HostPlacement {
            player: "bob".into(),
        };
        let rep_plc = ReplicatedPlacement {
            players: ["alice".into(), "bob".into(), "carole".into()],
        };

        let ctx = SymbolicContext::default();
        let x: Symbolic<Ring64Tensor> = alice_plc.sample(&ctx);
        let y: Symbolic<Ring64Tensor> = bob_plc.sample(&ctx);
        let xe = rep_plc.share(&ctx, &x);
        let ye = rep_plc.share(&ctx, &y);
        let ze = rep_plc.add(&ctx, &xe, &ye);
        let z = bob_plc.reveal(&ctx, &ze);
        println!("SYMBOLIC {:?}", z);
    }

    #[test]
    fn test_rep_addsymbolic() {
        let alice_plc = HostPlacement {
            player: "alice".into(),
        };
        let bob_plc = HostPlacement {
            player: "bob".into(),
        };
        let rep_plc = ReplicatedPlacement {
            players: ["alice".into(), "bob".into(), "carole".into()],
        };

        let ctx = SymbolicContext::default();
        let x: Symbolic<Ring64Tensor> = alice_plc.sample(&ctx);
        let y: Symbolic<Ring64Tensor> = bob_plc.sample(&ctx);
        let xe = rep_plc.share(&ctx, &x);
        let ze = rep_plc.add(&ctx, &y, &xe);
        println!("SYMBOLIC {:?}", ze);
    }

    #[test]
    fn test_fixed_add() {
        let alice = HostPlacement {
            player: "alice".into(),
        };
        let bob = HostPlacement {
            player: "bob".into(),
        };
        let rep = ReplicatedPlacement {
            players: ["alice".into(), "bob".into(), "carole".into()],
        };

        let x = Fixed64Tensor::RingTensor(RingTensor(5 * 256, alice.clone()));
        let y = Fixed64Tensor::RingTensor(RingTensor(7 * 256, bob.clone()));

        let ctx = ConcreteContext::default();
        let z = rep.add(&ctx, &x, &y);

        println!("{:?}", z);
    }

    #[test]
    fn test_fixed_add_symb() {
        let alice = HostPlacement {
            player: "alice".into(),
        };
        let bob = HostPlacement {
            player: "bob".into(),
        };
        let rep = ReplicatedPlacement {
            players: ["alice".into(), "bob".into(), "carole".into()],
        };

        let x: <Fixed128Tensor as KnownType>::Symbolic = Symbolic::Symbolic(SymbolicHandle {
            op: "x".into(),
            plc: alice.clone().into(),
        });
        let y: <Fixed128Tensor as KnownType>::Symbolic = Symbolic::Symbolic(SymbolicHandle {
            op: "y".into(),
            plc: bob.clone().into(),
        });

        let ctx = SymbolicContext::default();
        let z = rep.add(&ctx, &x, &y);

        println!("{:?}", z);

        let ops = ctx.ops.read().unwrap();
        for op in ops.iter() {
            println!("  {:?}", op);
        }
    }

    #[test]
    fn test_fixed_add_symb_lower() {
        let alice = HostPlacement {
            player: "alice".into(),
        };
        let bob = HostPlacement {
            player: "bob".into(),
        };
        let rep = ReplicatedPlacement {
            players: ["alice".into(), "bob".into(), "carole".into()],
        };

        let x: <Fixed64Tensor as KnownType>::Symbolic = Symbolic::Concrete(
            FixedTensor::RingTensor(Symbolic::Symbolic(SymbolicHandle {
                op: "x".into(),
                plc: alice.clone(),
            })),
        );
        let y: <Fixed64Tensor as KnownType>::Symbolic = Symbolic::Concrete(
            FixedTensor::RingTensor(Symbolic::Symbolic(SymbolicHandle {
                op: "y".into(),
                plc: bob.clone(),
            })),
        );

        let ctx = SymbolicContext::default();
        let z = rep.add(&ctx, &x, &y);

        println!("{:?}", z);

        let ops = ctx.ops.read().unwrap();
        for op in ops.iter() {
            println!("  {:?}", op);
        }
    }

    #[test]
    fn test_rep_exec() {
        #![allow(clippy::redundant_clone)]

        use std::collections::HashMap;

        let alice_plc = HostPlacement {
            player: "alice".into(),
        };
        let bob_plc = HostPlacement {
            player: "bob".into(),
        };
        let rep_plc = ReplicatedPlacement {
            players: ["alice".into(), "bob".into(), "carole".into()],
        };

        let ops: Vec<Operation> = vec![
            Operation {
                name: "x".into(),
                operator: RingSampleOp {
                    sig: NullarySignature {
                        ret: Ty::Ring128Tensor,
                    }
                    .into(),
                }
                .into(),
                operands: vec![],
                plc: alice_plc.clone().into(),
            },
            Operation {
                name: "xe".into(),
                operator: RepShareOp {
                    sig: UnarySignature {
                        arg0: Ty::Ring128Tensor,
                        ret: Ty::Replicated128Tensor,
                    }
                    .into(),
                }
                .into(),
                operands: vec!["x".into()],
                plc: rep_plc.clone().into(),
            },
            Operation {
                name: "y".into(),
                operator: RingSampleOp {
                    sig: NullarySignature {
                        ret: Ty::Ring128Tensor,
                    }
                    .into(),
                }
                .into(),
                operands: vec![],
                plc: bob_plc.clone().into(),
            },
            Operation {
                name: "ye".into(),
                operator: RepShareOp {
                    sig: UnarySignature {
                        arg0: Ty::Ring128Tensor,
                        ret: Ty::Replicated128Tensor,
                    }
                    .into(),
                }
                .into(),
                operands: vec!["y".into()],
                plc: rep_plc.clone().into(),
            },
            Operation {
                name: "s".into(),
                operator: RepSetupOp {
                    sig: NullarySignature {
                        ret: Ty::ReplicatedSetup,
                    }
                    .into(),
                }
                .into(),
                operands: vec![],
                plc: rep_plc.clone().into(),
            },
            Operation {
                name: "ze".into(),
                operator: RepMulOp {
                    sig: TernarySignature {
                        arg0: Ty::ReplicatedSetup,
                        arg1: Ty::Replicated128Tensor,
                        arg2: Ty::Replicated128Tensor,
                        ret: Ty::Replicated128Tensor,
                    }
                    .into(),
                }
                .into(),
                operands: vec!["s".into(), "xe".into(), "ye".into()],
                plc: rep_plc.clone().into(),
            },
            Operation {
                name: "ve".into(),
                operator: RepMulOp {
                    sig: TernarySignature {
                        arg0: Ty::ReplicatedSetup,
                        arg1: Ty::Replicated128Tensor,
                        arg2: Ty::Replicated128Tensor,
                        ret: Ty::Replicated128Tensor,
                    }
                    .into(),
                }
                .into(),
                operands: vec!["s".into(), "xe".into(), "ye".into()],
                plc: rep_plc.clone().into(),
            },
        ];

        let ctx = SymbolicContext::default();
        let mut env: HashMap<String, SymbolicValue> = HashMap::default();

        for op in ops.iter() {
            let operator = op.operator.clone();
            let operands = op
                .operands
                .iter()
                .map(|input_name| env.get(input_name).unwrap().clone())
                .collect();
            let res = ctx.execute(operator, &op.plc, operands);
            env.insert(op.name.clone(), res);
        }

        println!("{:?}\n\n", env);

        let replicated_keys = HashMap::new();
        let ctx = ConcreteContext { replicated_keys };

        let mut env: HashMap<String, Value> = HashMap::default();

        for op in ops.iter() {
            let operator = op.operator.clone();
            let operands = op
                .operands
                .iter()
                .map(|input_name| env.get(input_name).unwrap().clone())
                .collect();
            let res = ctx.execute(operator, &op.plc, operands);
            env.insert(op.name.clone(), res);
        }

        println!("{:?}", env);

        // let ops = ctx.ops.read().unwrap();
        // for op in ops.iter() {
        //     println!("  {:?}", op);
        // }

        // let comp = r#"

        // "#.try_into().unwrap();

        // let exec = SymbolicExecutor;
        // exec.eval(comp);
    }

    #[test]
    fn test_rep_bin_exec() {
        #![allow(clippy::redundant_clone)]

        use std::collections::HashMap;

        let alice_plc = HostPlacement {
            player: "alice".into(),
        };
        let bob_plc = HostPlacement {
            player: "bob".into(),
        };
        let rep_plc = ReplicatedPlacement {
            players: ["alice".into(), "bob".into(), "carole".into()],
        };

        let ops: Vec<Operation> = vec![
            Operation {
                name: "x".into(),
                operator: BitSampleOp {
                    sig: NullarySignature { ret: Ty::BitTensor }.into(),
                }
                .into(),
                operands: vec![],
                plc: alice_plc.clone().into(),
            },
            Operation {
                name: "xe".into(),
                operator: RepShareOp {
                    sig: UnarySignature {
                        arg0: Ty::BitTensor,
                        ret: Ty::ReplicatedBitTensor,
                    }
                    .into(),
                }
                .into(),
                operands: vec!["x".into()],
                plc: rep_plc.clone().into(),
            },
            Operation {
                name: "y".into(),
                operator: BitSampleOp {
                    sig: NullarySignature { ret: Ty::BitTensor }.into(),
                }
                .into(),
                operands: vec![],
                plc: bob_plc.clone().into(),
            },
            Operation {
                name: "ye".into(),
                operator: RepShareOp {
                    sig: UnarySignature {
                        arg0: Ty::BitTensor,
                        ret: Ty::ReplicatedBitTensor,
                    }
                    .into(),
                }
                .into(),
                operands: vec!["y".into()],
                plc: rep_plc.clone().into(),
            },
            Operation {
                name: "s".into(),
                operator: RepSetupOp {
                    sig: NullarySignature {
                        ret: Ty::ReplicatedSetup,
                    }
                    .into(),
                }
                .into(),
                operands: vec![],
                plc: rep_plc.clone().into(),
            },
            Operation {
                name: "ze".into(),
                operator: RepMulOp {
                    sig: TernarySignature {
                        arg0: Ty::ReplicatedSetup,
                        arg1: Ty::ReplicatedBitTensor,
                        arg2: Ty::ReplicatedBitTensor,
                        ret: Ty::ReplicatedBitTensor,
                    }
                    .into(),
                }
                .into(),
                operands: vec!["s".into(), "xe".into(), "ye".into()],
                plc: rep_plc.clone().into(),
            },
            Operation {
                name: "ve".into(),
                operator: RepMulOp {
                    sig: TernarySignature {
                        arg0: Ty::ReplicatedSetup,
                        arg1: Ty::ReplicatedBitTensor,
                        arg2: Ty::ReplicatedBitTensor,
                        ret: Ty::ReplicatedBitTensor,
                    }
                    .into(),
                }
                .into(),
                operands: vec!["s".into(), "xe".into(), "ye".into()],
                plc: rep_plc.clone().into(),
            },
        ];

        let ctx = SymbolicContext::default();
        let mut env: HashMap<String, SymbolicValue> = HashMap::default();

        for op in ops.iter() {
            let operator = op.operator.clone();
            let operands = op
                .operands
                .iter()
                .map(|input_name| env.get(input_name).unwrap().clone())
                .collect();
            let res = ctx.execute(operator, &op.plc, operands);
            env.insert(op.name.clone(), res);
        }

        println!("{:?}", env);

        let ctx = ConcreteContext::default();
        let mut env: HashMap<String, Value> = HashMap::default();

        for op in ops.iter() {
            let operator = op.operator.clone();
            let operands = op
                .operands
                .iter()
                .map(|input_name| env.get(input_name).unwrap().clone())
                .collect();
            let res = ctx.execute(operator, &op.plc, operands);
            env.insert(op.name.clone(), res);
        }

        println!("{:?}", env);
    }

    #[test]
    fn test_add_exec() {
        let alice = HostPlacement {
            player: "alice".into(),
        };
        let bob = HostPlacement {
            player: "bob".into(),
        };
        let add_plc = AdditivePlacement {
            players: ["alice".into(), "bob".into()],
        };

        let x = Additive64Tensor {
            shares: [RingTensor(1, alice.clone()), RingTensor(2, bob.clone())],
        };
        let y = Additive64Tensor {
            shares: [RingTensor(1, alice.clone()), RingTensor(2, bob.clone())],
        };

        let ctx = ConcreteContext::default();
        let z = add_plc.add(&ctx, &x, &y);
        let z_reveal = alice.reveal(&ctx, &z);
        println!("{:?}", z_reveal);
        // TODO: fix this after placement merge
        // assert_eq!(z_reveal, RingTensor(6, alice.clone()));

        let z2 = add_plc.mul(&ctx, &x, &RingTensor(10, bob.clone()));
        let z2_reveal = bob.reveal(&ctx, &z2);

        assert_eq!(z2_reveal, RingTensor(30, bob.clone()));
    }
}
