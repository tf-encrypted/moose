#![allow(dead_code)]
#![allow(unused_variables)]

use macros::with_context;
use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::ops::{Add, Mul, Neg, Shl, Shr, Sub};
use std::ops::{BitAnd, BitXor};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Ty {
    Fixed64Tensor,
    Fixed128Tensor,
    BitTensor,
    Ring32Tensor,
    Ring64Tensor,
    Ring128Tensor,
    Replicated64Tensor,
    Replicated128Tensor,
    ReplicatedBitTensor,
    Additive64Tensor,
    Additive128Tensor,
    ReplicatedSetup,
    PrfKey,
    Shape,
    Ring64,
    Ring128,
    Bit,
}

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
            Ty::Bit => SymbolicValue::Bit(Symbolic::Symbolic(SymbolicHandle {
                op: op_name.into(),
                plc: plc.try_into().unwrap(),
            })),
            Ty::Ring32Tensor => SymbolicValue::Ring32Tensor(Symbolic::Symbolic(SymbolicHandle {
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
            Ty::Ring64 => SymbolicValue::Ring64(Symbolic::Symbolic(SymbolicHandle {
                op: op_name.into(),
                plc: plc.try_into().unwrap(),
            })),
            Ty::Ring128 => SymbolicValue::Ring128(Symbolic::Symbolic(SymbolicHandle {
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
            Ty::Shape => SymbolicValue::Shape(Symbolic::Symbolic(SymbolicHandle {
                op: op_name.into(),
                plc: plc.try_into().unwrap(),
            })),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Fixed64Tensor(Fixed64Tensor),
    Fixed128Tensor(Fixed128Tensor),
    BitTensor(BitTensor),
    Ring32Tensor(Ring32Tensor),
    Ring64Tensor(Ring64Tensor),
    Ring128Tensor(Ring128Tensor),
    Replicated64Tensor(Replicated64Tensor),
    Replicated128Tensor(Replicated128Tensor),
    ReplicatedBitTensor(ReplicatedBitTensor),
    Additive64Tensor(Additive64Tensor),
    Additive128Tensor(Additive128Tensor),
    ReplicatedSetup(ReplicatedSetup),
    PrfKey(PrfKey),
    Shape(Shape),
    Ring64(Ring64),
    Ring128(Ring128),
    Bit(Bit),
}

impl Value {
    pub fn ty(&self) -> Ty {
        match self {
            Value::Fixed64Tensor(_) => Ty::Fixed64Tensor,
            Value::Fixed128Tensor(_) => Ty::Fixed128Tensor,
            Value::BitTensor(_) => Ty::BitTensor,
            Value::Ring32Tensor(_) => Ty::Ring32Tensor,
            Value::Ring64Tensor(_) => Ty::Ring64Tensor,
            Value::Ring128Tensor(_) => Ty::Ring128Tensor,
            Value::Replicated64Tensor(_) => Ty::Replicated64Tensor,
            Value::Replicated128Tensor(_) => Ty::Replicated128Tensor,
            Value::ReplicatedBitTensor(_) => Ty::ReplicatedBitTensor,
            Value::Additive64Tensor(_) => Ty::Additive64Tensor,
            Value::Additive128Tensor(_) => Ty::Additive128Tensor,
            Value::ReplicatedSetup(_) => Ty::ReplicatedSetup,
            Value::PrfKey(_) => Ty::PrfKey,
            Value::Shape(_) => Ty::Shape,
            Value::Ring64(_) => Ty::Ring64,
            Value::Ring128(_) => Ty::Ring128,
            Value::Bit(_) => Ty::Bit,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SymbolicValue {
    Fixed64Tensor(<Fixed64Tensor as KnownType>::Symbolic),
    Fixed128Tensor(<Fixed128Tensor as KnownType>::Symbolic),
    BitTensor(<BitTensor as KnownType>::Symbolic),
    Bit(<Bit as KnownType>::Symbolic),
    Ring32Tensor(<Ring32Tensor as KnownType>::Symbolic),
    Ring64Tensor(<Ring64Tensor as KnownType>::Symbolic),
    Ring128Tensor(<Ring128Tensor as KnownType>::Symbolic),
    Replicated64Tensor(<Replicated64Tensor as KnownType>::Symbolic),
    Replicated128Tensor(<Replicated128Tensor as KnownType>::Symbolic),
    ReplicatedBitTensor(<ReplicatedBitTensor as KnownType>::Symbolic),
    Additive64Tensor(<Additive64Tensor as KnownType>::Symbolic),
    Additive128Tensor(<Additive128Tensor as KnownType>::Symbolic),
    ReplicatedSetup(<ReplicatedSetup as KnownType>::Symbolic),
    PrfKey(<PrfKey as KnownType>::Symbolic),
    Shape(<Shape as KnownType>::Symbolic),
    Ring64(<Ring64 as KnownType>::Symbolic),
    Ring128(<Ring128 as KnownType>::Symbolic),
}

macro_rules! value {
    ($t:ident, $st:ty) => {
        impl From<$t> for Value {
            fn from(x: $t) -> Value {
                Value::$t(x)
            }
        }

        impl From<&$t> for Value {
            fn from(x: &$t) -> Value {
                Value::$t(x.clone())
            }
        }

        impl TryFrom<Value> for $t {
            type Error = ();

            fn try_from(x: Value) -> Result<Self, Self::Error> {
                match x {
                    Value::$t(x) => Ok(x),
                    _ => Err(()),
                }
            }
        }

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
            const TY: Ty = Ty::$t;
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
    Fixed64Tensor,
    Symbolic<
        FixedTensor<
            <Ring64Tensor as KnownType>::Symbolic,
            <Replicated64Tensor as KnownType>::Symbolic,
        >,
    >
);
value!(
    Fixed128Tensor,
    Symbolic<
        FixedTensor<
            <Ring128Tensor as KnownType>::Symbolic,
            <Replicated128Tensor as KnownType>::Symbolic,
        >,
    >
);
value!(BitTensor, Symbolic<BitTensor>);
value!(Ring32Tensor, Symbolic<Ring32Tensor>);
value!(Ring64Tensor, Symbolic<Ring64Tensor>);
value!(Ring128Tensor, Symbolic<Ring128Tensor>);
value!(
    Replicated64Tensor,
    Symbolic<ReplicatedTensor<<Ring64Tensor as KnownType>::Symbolic>>
);
value!(
    Replicated128Tensor,
    Symbolic<ReplicatedTensor<<Ring128Tensor as KnownType>::Symbolic>>
);
value!(
    ReplicatedBitTensor,
    Symbolic<ReplicatedTensor<Symbolic<BitTensor>>>
);
value!(
    Additive64Tensor,
    Symbolic<AdditiveTensor<<Ring64Tensor as KnownType>::Symbolic>>
);
value!(
    Additive128Tensor,
    Symbolic<AdditiveTensor<<Ring128Tensor as KnownType>::Symbolic>>
);
value!(
    ReplicatedSetup,
    Symbolic<AbstractReplicatedSetup<<PrfKey as KnownType>::Symbolic>>
);
value!(PrfKey, Symbolic<PrfKey>);
value!(Shape, Symbolic<Shape>);
value!(Ring64, Symbolic<Ring64>);
value!(Ring128, Symbolic<Ring128>);
value!(Bit, Symbolic<Bit>);

#[derive(Clone, Debug, PartialEq)]
pub enum Symbolic<T: Placed> {
    Symbolic(SymbolicHandle<T::Placement>),
    Concrete(T),
}

impl Placed for Shape {
    type Placement = HostPlacement;

    fn placement(&self) -> Self::Placement {
        self.1.clone()
    }
}

impl<RingTensorT, ReplicatedTensorT> Placed for FixedTensor<RingTensorT, ReplicatedTensorT>
where
    RingTensorT: Placed,
    RingTensorT::Placement: Into<Placement>,
    ReplicatedTensorT: Placed,
    ReplicatedTensorT::Placement: Into<Placement>,
{
    type Placement = Placement;

    fn placement(&self) -> Self::Placement {
        match self {
            FixedTensor::RingTensor(x) => x.placement().into(),
            FixedTensor::ReplicatedTensor(x) => x.placement().into(),
        }
    }
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

impl TryFrom<Symbolic<Shape>> for Shape {
    type Error = Symbolic<Self>;

    fn try_from(x: Symbolic<Shape>) -> Result<Self, Self::Error> {
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

impl<R> From<Ring<R>> for Symbolic<Ring<R>>
where
    R: Placed<Placement = HostPlacement>,
{
    fn from(x: Ring<R>) -> Self {
        Symbolic::Concrete(x)
    }
}

impl From<Shape> for Symbolic<Shape> {
    fn from(x: Shape) -> Self {
        Symbolic::Concrete(x)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ring<T>(T, HostPlacement);

impl BitTensor {
    fn fill(el: u8, plc: HostPlacement) -> BitTensor {
        BitTensor(el, plc)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BitTensor(u8, HostPlacement);

impl BitXor for BitTensor {
    type Output = BitTensor;
    fn bitxor(self, other: Self) -> Self::Output {
        BitTensor(self.0 ^ other.0, self.1)
    }
}

impl BitAnd for BitTensor {
    type Output = BitTensor;
    fn bitand(self, other: Self) -> Self::Output {
        BitTensor(self.0 & other.0, self.1)
    }
}

pub type Ring64 = Ring<u64>;

pub type Ring128 = Ring<u128>;

pub type Bit = Ring<u8>;

pub type Fixed64Tensor = FixedTensor<Ring64Tensor, Replicated64Tensor>;

pub type Fixed128Tensor = FixedTensor<Ring128Tensor, Replicated128Tensor>;

#[derive(Clone, Debug, PartialEq)]
pub enum FixedTensor<RingTensorT, ReplicatedTensorT> {
    RingTensor(RingTensorT),
    ReplicatedTensor(ReplicatedTensorT),
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
            Operator::ShapeOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RingSampleOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::BitSampleOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RingAddOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::BitXorOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::BitAndOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RingSubOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RingNegOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RingMulOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RingShlOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RingShrOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::FillOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::OnesOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RepSetupOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RepShareOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RepRevealOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RepAddOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RepMulOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RepToAddOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::RepTruncPrOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::AddToRepOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::AdditiveAddOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
            Operator::AdditiveSubOp(op) => DispatchKernel::compile(&op, ctx, plc)(operands),
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

trait PlacementBitCompose<C: Context, R> {
    fn bit_compose(&self, ctx: &C, bits: &[R]) -> R;
}

impl<C: Context, R> PlacementBitCompose<C, R> for HostPlacement
where
    R: Clone,
    HostPlacement: PlacementShl<C, R, R>,
    HostPlacement: PlacementTreeReduce<C, R>,
{
    fn bit_compose(&self, ctx: &C, bits: &[R]) -> R {
        let shifted_bits: Vec<_> = (0..bits.len())
            .map(|i| self.shl(ctx, i, &bits[i]))
            .collect();
        self.tree_reduce(ctx, &shifted_bits)
    }
}

trait PlacementTreeReduce<C: Context, R> {
    fn tree_reduce(&self, ctx: &C, sequence: &[R]) -> R;
}

impl<C: Context, R> PlacementTreeReduce<C, R> for HostPlacement
where
    R: Clone,
    HostPlacement: PlacementAdd<C, R, R, R>,
{
    fn tree_reduce(&self, ctx: &C, sequence: &[R]) -> R {
        let n = sequence.len();
        if n == 1 {
            sequence[0].clone()
        } else {
            let mut reduced: Vec<_> = (0..n / 2)
                .map(|i| {
                    let x0: &R = &sequence[2 * i];
                    let x1: &R = &sequence[2 * i + 1];
                    self.add(ctx, &x0, &x1)
                })
                .collect();
            if n % 2 == 1 {
                reduced.push(sequence[n - 1].clone());
            }
            self.tree_reduce(ctx, &reduced)
        }
    }
}

trait PlacementArithmeticXor<C: Context, R> {
    fn arithmetic_xor(&self, ctx: &C, x: &AdditiveTensor<R>, y: &R) -> AdditiveTensor<R>;
    // compute x + y - 2 * x * y
}

impl<C: Context, R> PlacementArithmeticXor<C, R> for AdditivePlacement
where
    AdditivePlacement: PlacementAdd<C, AdditiveTensor<R>, R, AdditiveTensor<R>>,
    AdditivePlacement: PlacementAdd<C, AdditiveTensor<R>, AdditiveTensor<R>, AdditiveTensor<R>>,
    AdditivePlacement: PlacementMul<C, AdditiveTensor<R>, R, AdditiveTensor<R>>,
    AdditivePlacement: PlacementSub<C, AdditiveTensor<R>, AdditiveTensor<R>, AdditiveTensor<R>>,
{
    fn arithmetic_xor(&self, ctx: &C, x: &AdditiveTensor<R>, y: &R) -> AdditiveTensor<R> {
        let sum = self.add(ctx, x, y);
        let (player_a, player_b) = self.host_placements();
        let local_prod = self.mul(ctx, x, y);
        let twice_prod = self.add(ctx, &local_prod, &local_prod);
        self.sub(ctx, &sum, &twice_prod)
    }
}
trait RingSize {
    const SIZE: usize;
}

impl<R: RingSize + Placed> RingSize for Symbolic<R> {
    const SIZE: usize = <R as RingSize>::SIZE;
}

impl RingSize for Ring64Tensor {
    const SIZE: usize = 64;
}

impl RingSize for Ring128Tensor {
    const SIZE: usize = 128;
}

trait PlacementTruncPrWithPrep<C: Context, R, K> {
    fn trunc_pr(
        &self,
        ctx: &C,
        x: &AdditiveTensor<R>,
        m: usize,
        provider: HostPlacement,
    ) -> AdditiveTensor<R>;
    fn get_prep(
        &self,
        ctx: &C,
        shape: &Shape,
        m: usize,
        provider: HostPlacement,
    ) -> (AdditiveTensor<R>, AdditiveTensor<R>, AdditiveTensor<R>);
}

impl<C: Context, R, K> PlacementTruncPrWithPrep<C, R, K> for AdditivePlacement
where
    R: RingSize,
    AdditivePlacement: PlacementAdd<C, AdditiveTensor<R>, AdditiveTensor<R>, AdditiveTensor<R>>,
    AdditivePlacement: PlacementAdd<C, R, AdditiveTensor<R>, AdditiveTensor<R>>,
    AdditivePlacement: PlacementAdd<C, AdditiveTensor<R>, R, AdditiveTensor<R>>,
    AdditivePlacement: PlacementArithmeticXor<C, R>,
    AdditivePlacement: PlacementFill<C, Shape, AdditiveTensor<R>>, // TODO: Fix shape; Use type parameter
    AdditivePlacement: PlacementMul<C, AdditiveTensor<R>, R, AdditiveTensor<R>>,
    AdditivePlacement: PlacementShl<C, AdditiveTensor<R>, AdditiveTensor<R>>,
    AdditivePlacement: PlacementSub<C, AdditiveTensor<R>, AdditiveTensor<R>, AdditiveTensor<R>>,
    AdditivePlacement: PlacementSub<C, AdditiveTensor<R>, R, AdditiveTensor<R>>,
    HostPlacement: PlacementBitCompose<C, R> + PlacementKeyGen<C, K> + PlacementSub<C, R, R, R>,
    HostPlacement: PlacementOnes<C, Shape, R>,
    HostPlacement: PlacementReveal<C, AdditiveTensor<R>, R>,
    HostPlacement: PlacementSample<C, R>,
    HostPlacement: PlacementShape<C, R, Shape>,
    HostPlacement: PlacementShl<C, R, R>,
    HostPlacement: PlacementShr<C, R, R>,
    R: Into<Value> + Clone,
{
    fn trunc_pr(
        &self,
        ctx: &C,
        x: &AdditiveTensor<R>,
        m: usize,
        third_party: HostPlacement,
    ) -> AdditiveTensor<R> {
        // consider input is always signed
        let (player_a, player_b) = self.host_placements();
        let AdditiveTensor { shares: [x0, x1] } = x;

        let k = R::SIZE - 1;
        // TODO(Dragos)this is optional if we work with unsigned numbers
        let x_shape = player_a.shape(ctx, x0);

        let ones = player_a.ones(ctx, &x_shape);

        let twok = player_a.shl(ctx, k, &ones);
        let positive = self.add(ctx, x, &twok);

        let (r, r_top, r_msb) = self.get_prep(ctx, &x_shape, m, third_party);

        let masked = self.add(ctx, &positive, &r);
        // (Dragos) Note that these opening should be done to all players for active security.
        let opened_masked_a = player_a.reveal(ctx, &masked);

        let no_msb_mask = player_a.shl(ctx, 1, &opened_masked_a);
        let opened_mask_tr = player_a.shr(ctx, m + 1, &no_msb_mask);

        let msb_mask = player_a.shr(ctx, R::SIZE - 1, &opened_masked_a);
        let msb_to_correct = self.arithmetic_xor(ctx, &r_msb, &msb_mask);
        let shifted_msb = self.shl(ctx, R::SIZE - 1 - m, &msb_to_correct);

        let output = self.add(ctx, &self.sub(ctx, &shifted_msb, &r_top), &opened_mask_tr);
        // TODO(Dragos)this is optional if we work with unsigned numbers
        let remainder = player_a.shl(ctx, k - 1 - m, &ones);
        self.sub(ctx, &output, &remainder)
    }
    fn get_prep(
        &self,
        ctx: &C,
        shape: &Shape,
        m: usize,
        provider: HostPlacement,
    ) -> (AdditiveTensor<R>, AdditiveTensor<R>, AdditiveTensor<R>) {
        let (player_a, player_b) = self.host_placements();

        let r_bits: Vec<_> = (0..R::SIZE).map(|_| provider.sample(ctx)).collect();
        let r = provider.bit_compose(ctx, &r_bits);

        let r_top_bits: Vec<_> = (m..R::SIZE - 1).map(|i| r_bits[i].clone()).collect();
        let r_top_ring = provider.bit_compose(ctx, &r_bits[m..R::SIZE - 1]);
        let r_msb = r_bits[R::SIZE - 1].clone();

        let tmp: [R; 3] = [r, r_top_ring, r_msb];

        let k = provider.keygen(ctx);
        let mut results = Vec::<AdditiveTensor<R>>::new();
        for item in &tmp {
            let share0 = provider.sample(ctx);
            let share1 = provider.sub(ctx, &item, &share0);
            // TODO(Dragos) this could probably be optimized by sending the key to p0
            results.push(AdditiveTensor {
                shares: [share0.clone(), share1.clone()],
            })
        }
        (results[0].clone(), results[1].clone(), results[2].clone())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepTruncPrOp {
    sig: Signature,
    amount: usize,
}

modelled!(PlacementTruncPr::trunc_pr, ReplicatedPlacement, attributes[amount: usize] (ReplicatedSetup, Replicated64Tensor) -> Replicated64Tensor, RepTruncPrOp);
modelled!(PlacementTruncPr::trunc_pr, ReplicatedPlacement, attributes[amount: usize] (ReplicatedSetup, Replicated128Tensor) -> Replicated128Tensor, RepTruncPrOp);

kernel! {
    RepTruncPrOp,
    [
        (ReplicatedPlacement,  (ReplicatedSetup, Replicated64Tensor) -> Replicated64Tensor => attributes[amount] Self::kernel),
        (ReplicatedPlacement,  (ReplicatedSetup, Replicated128Tensor) -> Replicated128Tensor => attributes[amount] Self::kernel),
    ]
}

impl RepTruncPrOp {
    fn kernel<C: Context, K, R>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        amount: usize,
        s: AbstractReplicatedSetup<K>,
        xe: ReplicatedTensor<R>,
    ) -> ReplicatedTensor<R>
    where
        AdditivePlacement: PlacementTruncPrWithPrep<C, R, K>
            + PlacementRepToAdd<C, ReplicatedTensor<R>, AdditiveTensor<R>>,
        ReplicatedPlacement: PlacementAddToRep<C, AdditiveTensor<R>, ReplicatedTensor<R>>,
    {
        let m = amount;

        let (player0, player1, player2) = rep.host_placements();
        let add_plc = AdditivePlacement {
            players: [player0.player, player1.player],
        };
        let x_add = add_plc.rep_to_add(ctx, &xe);
        let x_trunc = add_plc.trunc_pr(ctx, &x_add, m, player2);
        rep.add_to_rep(ctx, &x_trunc)
    }
}

impl FillOp {
    fn kernel64<C: Context>(
        ctx: &C,
        plc: &HostPlacement,
        value: Value,
        shape: Shape,
    ) -> Ring64Tensor {
        // TODO: Pass in typed value instead of Value
        match value {
            Value::Ring64(el) => Ring64Tensor::fill(el.0, plc.clone()),
            _ => unimplemented!(), // ok
        }
    }

    fn additive_kernel64<C: Context>(
        ctx: &C,
        plc: &AdditivePlacement,
        value: Value,
        shape: Shape,
    ) -> Additive64Tensor {
        // TODO: Pass in typed value instead of Value
        // This should be PublicTensor
        let (player_a, player_b) = plc.host_placements();
        match value {
            Value::Ring64(el) => {
                let shares = [
                    Ring64Tensor::fill(el.0, player_a),
                    Ring64Tensor::fill(0, player_b),
                ];
                AdditiveTensor { shares }
            }
            _ => unimplemented!(), // ok
        }
    }

    fn kernel128<C: Context>(
        ctx: &C,
        plc: &HostPlacement,
        value: Value,
        shape: Shape,
    ) -> Ring128Tensor {
        // TODO: Pass in typed value instead of Value
        match value {
            Value::Ring128(el) => Ring128Tensor::fill(el.0, plc.clone()),
            _ => unimplemented!(), // ok
        }
    }

    fn additive_kernel128<C: Context>(
        ctx: &C,
        plc: &AdditivePlacement,
        value: Value,
        shape: Shape,
    ) -> Additive128Tensor {
        // TODO: Pass in typed value instead of Value
        // This should be PublicTensor
        let (player_a, player_b) = plc.host_placements();
        match value {
            Value::Ring128(el) => {
                let shares = [
                    Ring128Tensor::fill(el.0, player_a),
                    Ring128Tensor::fill(0, player_b),
                ];
                AdditiveTensor { shares }
            }
            _ => unimplemented!(), // ok
        }
    }

    fn kernel8<C: Context>(ctx: &C, plc: &HostPlacement, value: Value, shape: Shape) -> BitTensor {
        // TODO: Pass in typed value instead of Value
        match value {
            Value::Bit(el) => {
                let val = el.0;
                assert!(
                    val == 0 || val == 1,
                    "cannot fill a BitTensor with a value {:?}",
                    val
                );
                BitTensor::fill(val, plc.clone())
            }
            _ => unimplemented!(), // ok
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RingShlOp {
    sig: Signature,
    amount: usize,
}

modelled!(PlacementShl::shl, HostPlacement, attributes[amount: usize] (Ring64Tensor) -> Ring64Tensor, RingShlOp);
modelled!(PlacementShl::shl, HostPlacement, attributes[amount: usize] (Ring128Tensor) -> Ring128Tensor, RingShlOp);
modelled!(PlacementShl::shl, AdditivePlacement, attributes[amount: usize] (Additive64Tensor) -> Additive64Tensor, RingShlOp);
modelled!(PlacementShl::shl, AdditivePlacement, attributes[amount: usize] (Additive128Tensor) -> Additive128Tensor, RingShlOp);

kernel! {
    RingShlOp,
    [
        (HostPlacement, (Ring64Tensor) -> Ring64Tensor => attributes[amount] Self::kernel),
        (HostPlacement, (Ring128Tensor) -> Ring128Tensor => attributes[amount] Self::kernel),
        (AdditivePlacement, (Additive64Tensor) -> Additive64Tensor => attributes[amount] Self::additive_kernel),
        (AdditivePlacement, (Additive128Tensor) -> Additive128Tensor => attributes[amount] Self::additive_kernel),
    ]
}

impl RingShlOp {
    fn kernel<C: Context, T>(
        _ctx: &C,
        _plc: &HostPlacement,
        amount: usize,
        x: RingTensor<T>,
    ) -> RingTensor<T>
    where
        RingTensor<T>: Shl<usize, Output = RingTensor<T>>,
    {
        x << amount
    }

    fn additive_kernel<C: Context, T>(
        _ctx: &C,
        _plc: &AdditivePlacement,
        amount: usize,
        x: AdditiveTensor<T>,
    ) -> AdditiveTensor<T>
    where
        T: Shl<usize, Output = T>,
    {
        let (player0, player1) = _plc.host_placements();
        let AdditiveTensor { shares: [x0, x1] } = x;
        let z0 = with_context!(player0, ctx, x0 << amount);
        let z1 = with_context!(player1, ctx, x1 << amount);
        AdditiveTensor { shares: [z0, z1] }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BitSampleOp {
    sig: Signature,
}

modelled!(PlacementSample::sample, HostPlacement, () -> BitTensor, BitSampleOp);

kernel! {
    BitSampleOp,
    [
        (HostPlacement, () -> BitTensor => Self::kernel),
    ]
}

impl BitSampleOp {
    fn kernel(ctx: &ConcreteContext, plc: &HostPlacement) -> BitTensor {
        // TODO
        BitTensor(0, plc.clone())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FixedMulOp {
    sig: Signature,
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
        HostPlacement: PlacementReveal<C, ReplicatedTensorT, RingTensorT>,
        HostPlacement: PlacementMul<C, RingTensorT, RingTensorT, RingTensorT>,
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
        ReplicatedPlacement: PlacementShare<C, RingTensorT, ReplicatedTensorT>,
        ReplicatedPlacement: PlacementMulSetup<
            C,
            C::ReplicatedSetup,
            ReplicatedTensorT,
            ReplicatedTensorT,
            ReplicatedTensorT,
        >,
        ReplicatedPlacement:
            PlacementAdd<C, ReplicatedTensorT, ReplicatedTensorT, ReplicatedTensorT>,
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

#[derive(Clone, Debug, PartialEq)]
pub struct FixedAddOp {
    sig: Signature,
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
        HostPlacement: PlacementReveal<C, ReplicatedTensorT, RingTensorT>,
        HostPlacement: PlacementAdd<C, RingTensorT, RingTensorT, RingTensorT>,
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
        ReplicatedPlacement: PlacementShare<C, RingTensorT, ReplicatedTensorT>,
        ReplicatedPlacement:
            PlacementAdd<C, ReplicatedTensorT, ReplicatedTensorT, ReplicatedTensorT>,
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
