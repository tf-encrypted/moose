#![allow(dead_code)]
#![allow(unused_variables)]

use macros::eval_with_context;
use std::convert::{TryFrom, TryInto};
use std::ops::{Add, Mul, Sub};
use std::ops::{BitAnd, BitXor};

#[derive(Debug, Clone, PartialEq)]
enum Placement {
    HostPlacement(HostPlacement),
    ReplicatedPlacement(ReplicatedPlacement),
}

impl Placement {
    pub fn ty(&self) -> PlacementTy {
        match self {
            Placement::HostPlacement(plc) => plc.ty(),
            Placement::ReplicatedPlacement(plc) => plc.ty(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct HostPlacement {
    player: String,
}

#[derive(Clone, Debug, PartialEq)]
struct ReplicatedPlacement {
    players: [String; 3],
}

impl ReplicatedPlacement {
    pub fn host_placements(&self) -> (HostPlacement, HostPlacement, HostPlacement) {
        let player0 = HostPlacement {
            player: self.players[0].clone(),
        };
        let player1 = HostPlacement {
            player: self.players[1].clone(),
        };
        let player2 = HostPlacement {
            player: self.players[2].clone(),
        };
        (player0, player1, player2)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlacementTy {
    HostTy,
    ReplicatedTy,
}

trait KnownPlacement {
    const TY: PlacementTy;

    fn ty(&self) -> PlacementTy {
        Self::TY
    }
}

impl KnownPlacement for HostPlacement {
    const TY: PlacementTy = PlacementTy::HostTy;
}

impl KnownPlacement for ReplicatedPlacement {
    const TY: PlacementTy = PlacementTy::ReplicatedTy;
}

macro_rules! placement {
    ($t:ident) => {
        impl From<$t> for Placement {
            fn from(x: $t) -> Placement {
                Placement::$t(x)
            }
        }

        impl From<&$t> for Placement {
            fn from(x: &$t) -> Placement {
                Placement::$t(x.clone())
            }
        }

        impl TryFrom<Placement> for $t {
            type Error = ();

            fn try_from(x: Placement) -> Result<Self, Self::Error> {
                match x {
                    Placement::$t(x) => Ok(x),
                    _ => Err(()),
                }
            }
        }
    };
}

placement!(HostPlacement);
placement!(ReplicatedPlacement);

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Ty {
    BitTensor,
    Ring32Tensor,
    Ring64Tensor,
    Ring128Tensor,
    Replicated64Tensor,
    Replicated128Tensor,
    ReplicatedBitTensor,
    ReplicatedSetup,
    PrfKey,
}

impl Ty {
    pub fn synthesize_symbolic_value<S: Into<String>>(&self, op_name: S) -> SymbolicValue {
        match &self {
            Ty::BitTensor => {
                SymbolicValue::BitTensor(Symbolic::Symbolic(SymbolicHandle { op: op_name.into() }))
            }
            Ty::Ring32Tensor => SymbolicValue::Ring32Tensor(Symbolic::Symbolic(SymbolicHandle {
                op: op_name.into(),
            })),
            Ty::Ring64Tensor => SymbolicValue::Ring64Tensor(Symbolic::Symbolic(SymbolicHandle {
                op: op_name.into(),
            })),
            Ty::Ring128Tensor => SymbolicValue::Ring128Tensor(Symbolic::Symbolic(SymbolicHandle {
                op: op_name.into(),
            })),
            Ty::Replicated64Tensor => {
                SymbolicValue::Replicated64Tensor(Symbolic::Symbolic(SymbolicHandle {
                    op: op_name.into(),
                }))
            }
            Ty::Replicated128Tensor => {
                SymbolicValue::Replicated128Tensor(Symbolic::Symbolic(SymbolicHandle {
                    op: op_name.into(),
                }))
            }
            Ty::ReplicatedBitTensor => {
                SymbolicValue::ReplicatedBitTensor(Symbolic::Symbolic(SymbolicHandle {
                    op: op_name.into(),
                }))
            }
            Ty::ReplicatedSetup => {
                SymbolicValue::ReplicatedSetup(Symbolic::Symbolic(SymbolicHandle {
                    op: op_name.into(),
                }))
            }
            Ty::PrfKey => {
                SymbolicValue::PrfKey(Symbolic::Symbolic(SymbolicHandle { op: op_name.into() }))
            }
        }
    }
}

pub trait KnownType {
    type Symbolic;
    const TY: Ty;
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    BitTensor(BitTensor),
    Ring32Tensor(Ring32Tensor),
    Ring64Tensor(Ring64Tensor),
    Ring128Tensor(Ring128Tensor),
    Replicated64Tensor(Replicated64Tensor),
    Replicated128Tensor(Replicated128Tensor),
    ReplicatedBitTensor(ReplicatedBitTensor),
    ReplicatedSetup(ReplicatedSetup),
    PrfKey(PrfKey),
}

impl Value {
    pub fn ty(&self) -> Ty {
        match self {
            Value::BitTensor(_) => Ty::BitTensor,
            Value::Ring32Tensor(_) => Ty::Ring32Tensor,
            Value::Ring64Tensor(_) => Ty::Ring64Tensor,
            Value::Ring128Tensor(_) => Ty::Ring128Tensor,
            Value::Replicated64Tensor(_) => Ty::Replicated64Tensor,
            Value::Replicated128Tensor(_) => Ty::Replicated128Tensor,
            Value::ReplicatedBitTensor(_) => Ty::ReplicatedBitTensor,
            Value::ReplicatedSetup(_) => Ty::ReplicatedSetup,
            Value::PrfKey(_) => Ty::PrfKey,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SymbolicValue {
    BitTensor(<BitTensor as KnownType>::Symbolic),
    Ring32Tensor(<Ring32Tensor as KnownType>::Symbolic),
    Ring64Tensor(<Ring64Tensor as KnownType>::Symbolic),
    Ring128Tensor(<Ring128Tensor as KnownType>::Symbolic),
    Replicated64Tensor(<Replicated64Tensor as KnownType>::Symbolic),
    Replicated128Tensor(<Replicated128Tensor as KnownType>::Symbolic),
    ReplicatedBitTensor(<ReplicatedBitTensor as KnownType>::Symbolic),
    ReplicatedSetup(<ReplicatedSetup as KnownType>::Symbolic),
    PrfKey(<PrfKey as KnownType>::Symbolic),
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
value!(BitTensor, Symbolic<BitTensor>);
value!(Ring32Tensor, Symbolic<Ring32Tensor>);
value!(Ring64Tensor, Symbolic<Ring64Tensor>);
value!(Ring128Tensor, Symbolic<Ring128Tensor>);
value!(
    Replicated64Tensor,
    Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>>
);
value!(
    Replicated128Tensor,
    Symbolic<ReplicatedTensor<Symbolic<Ring128Tensor>>>
);
value!(
    ReplicatedBitTensor,
    Symbolic<ReplicatedTensor<Symbolic<BitTensor>>>
);
value!(
    ReplicatedSetup,
    Symbolic<AbstractReplicatedSetup<Symbolic<PrfKey>>>
);
value!(PrfKey, Symbolic<PrfKey>);

#[derive(Clone, Debug, PartialEq)]
pub enum Symbolic<T> {
    Symbolic(SymbolicHandle),
    Concrete(T),
}

#[derive(Clone, Debug, PartialEq)]
pub struct SymbolicHandle {
    op: String,
}

impl<T> From<SymbolicHandle> for Symbolic<T> {
    fn from(x: SymbolicHandle) -> Symbolic<T> {
        Symbolic::Symbolic(x)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Operator {
    PrfKeyGenOp(PrfKeyGenOp),
    RingAddOp(RingAddOp),
    BitXorOp(BitXorOp),
    BitAndOp(BitAndOp),
    RingSubOp(RingSubOp),
    RingMulOp(RingMulOp),
    RingSampleOp(RingSampleOp),
    BitSampleOp(BitSampleOp),
    RepSetupOp(RepSetupOp),
    RepAddOp(RepAddOp),
    RepMulOp(RepMulOp),
    RepShareOp(RepShareOp),
    RepRevealOp(RepRevealOp),
    ConstantOp(ConstantOp),
}

macro_rules! operator {
    ($t:ident) => {
        impl From<$t> for Operator {
            fn from(x: $t) -> Operator {
                Operator::$t(x)
            }
        }
    };
}

// NOTE a future improvement might be to have a single `operators!` macro
// that takes care of everything, including generating `enum Operator`.
operator!(PrfKeyGenOp);
operator!(RingAddOp);
operator!(BitXorOp);
operator!(BitAndOp);
operator!(RingSubOp);
operator!(RingMulOp);
operator!(RingSampleOp);
operator!(BitSampleOp);
operator!(RepSetupOp);
operator!(RepAddOp);
operator!(RepMulOp);
operator!(RepShareOp);
operator!(RepRevealOp);
operator!(ConstantOp);

#[derive(Clone, Debug, PartialEq)]
struct Operation {
    name: String,
    operator: Operator,
    operands: Vec<String>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Signature {
    Nullary(NullarySignature),
    Unary(UnarySignature),
    Binary(BinarySignature),
    Ternary(TernarySignature),
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct NullarySignature {
    ret: Ty,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct UnarySignature {
    arg0: Ty,
    ret: Ty,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BinarySignature {
    arg0: Ty,
    arg1: Ty,
    ret: Ty,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TernarySignature {
    arg0: Ty,
    arg1: Ty,
    arg2: Ty,
    ret: Ty,
}

impl From<NullarySignature> for Signature {
    fn from(s: NullarySignature) -> Signature {
        Signature::Nullary(s)
    }
}

impl From<UnarySignature> for Signature {
    fn from(s: UnarySignature) -> Signature {
        Signature::Unary(s)
    }
}

impl From<BinarySignature> for Signature {
    fn from(s: BinarySignature) -> Signature {
        Signature::Binary(s)
    }
}

impl From<TernarySignature> for Signature {
    fn from(s: TernarySignature) -> Signature {
        Signature::Ternary(s)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RingTensor<T>(T);

impl Add<RingTensor<u64>> for RingTensor<u64> {
    type Output = RingTensor<u64>;

    fn add(self, other: RingTensor<u64>) -> Self::Output {
        RingTensor(self.0.wrapping_add(other.0))
    }
}

impl Add<RingTensor<u128>> for RingTensor<u128> {
    type Output = RingTensor<u128>;

    fn add(self, other: RingTensor<u128>) -> Self::Output {
        RingTensor(self.0.wrapping_add(other.0))
    }
}

impl Sub<RingTensor<u64>> for RingTensor<u64> {
    type Output = RingTensor<u64>;

    fn sub(self, other: RingTensor<u64>) -> Self::Output {
        RingTensor(self.0.wrapping_sub(other.0))
    }
}

impl Sub<RingTensor<u128>> for RingTensor<u128> {
    type Output = RingTensor<u128>;

    fn sub(self, other: RingTensor<u128>) -> Self::Output {
        RingTensor(self.0.wrapping_sub(other.0))
    }
}

impl Mul<RingTensor<u64>> for RingTensor<u64> {
    type Output = RingTensor<u64>;

    fn mul(self, other: RingTensor<u64>) -> Self::Output {
        RingTensor(self.0.wrapping_mul(other.0))
    }
}

impl Mul<RingTensor<u128>> for RingTensor<u128> {
    type Output = RingTensor<u128>;

    fn mul(self, other: RingTensor<u128>) -> Self::Output {
        RingTensor(self.0.wrapping_mul(other.0))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BitTensor(u8);

impl BitXor for BitTensor {
    type Output = BitTensor;
    fn bitxor(self, other: Self) -> Self::Output {
        BitTensor(self.0 ^ other.0)
    }
}

impl BitAnd for BitTensor {
    type Output = BitTensor;
    fn bitand(self, other: Self) -> Self::Output {
        BitTensor(self.0 & other.0)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReplicatedTensor<R> {
    shares: [[R; 2]; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct PrfKey([u8; 16]);

#[derive(Clone, Debug, PartialEq)]
pub struct AbstractReplicatedSetup<K> {
    keys: [[K; 2]; 3],
}

#[derive(Clone, Debug, PartialEq)]
struct ReplicatedZeroShare<R> {
    alphas: [R; 3],
}

pub type Ring32Tensor = RingTensor<u32>;

pub type Ring64Tensor = RingTensor<u64>;

pub type Ring128Tensor = RingTensor<u128>;

pub type Replicated64Tensor = ReplicatedTensor<Ring64Tensor>;

pub type Replicated128Tensor = ReplicatedTensor<Ring128Tensor>;

pub type ReplicatedBitTensor = ReplicatedTensor<BitTensor>;

pub type ReplicatedSetup = AbstractReplicatedSetup<PrfKey>;

macro_rules! modelled {
    /*
    Nullary
    */
    ($t:ident, $plc:ty, () -> $u:ty, $op:ident) => {
        impl NullaryKernelCheck<ConcreteContext, $plc, $u> for $op {
            fn check(ctx: &ConcreteContext, plc: &$plc) -> $u {
                // NOTE we shouldn't do anything here, the kernel call is simply to check

                // NOTE not sure whether to add `unimplemented!`. it might be better to
                // simply make sure the Check traits are private.
                <Self as NullaryKernel<ConcreteContext, $plc, $u>>::kernel(ctx, plc)
            }
        }

        impl $t<ConcreteContext, $u> for $plc {
            fn apply(&self, ctx: &ConcreteContext) -> $u {
                let sig = NullarySignature {
                    ret: <$u as KnownType>::TY,
                };
                let op = $op::from_placement_signature(&self, sig);
                ctx.execute(op.into(), vec![]).try_into().unwrap()
            }
        }

        impl $t<SymbolicContext, <$u as KnownType>::Symbolic> for $plc {
            fn apply(&self, ctx: &SymbolicContext) -> <$u as KnownType>::Symbolic {
                let sig = NullarySignature {
                    ret: <$u as KnownType>::TY,
                };
                let op = $op::from_placement_signature(&self, sig);
                ctx.execute(op.into(), vec![]).try_into().unwrap()
            }
        }
    };

    /*
    Unary
    */
    ($t:ident, $plc:ty, ($t0:ty) -> $u:ty, $op:ident) => {
        impl UnaryKernelCheck<ConcreteContext, $plc, $t0, $u> for $op {
            fn check(ctx: &ConcreteContext, plc: &$plc, x0: $t0) -> $u {
                <Self as UnaryKernel<ConcreteContext, $plc, $t0, $u>>::kernel(ctx, plc, x0)
            }
        }

        impl $t<ConcreteContext, $t0> for $plc {
            type Output = $u;

            fn apply(&self, ctx: &ConcreteContext, x0: &$t0) -> Self::Output {
                let sig = UnarySignature {
                    arg0: <$t0 as KnownType>::TY,
                    ret: <$u as KnownType>::TY,
                };
                let op = $op::from_placement_signature(&self, sig);
                ctx.execute(op.into(), vec![x0.clone().into()])
                    .try_into()
                    .unwrap()
            }
        }

        impl $t<SymbolicContext, <$t0 as KnownType>::Symbolic> for $plc {
            type Output = <$u as KnownType>::Symbolic;

            fn apply(
                &self,
                ctx: &SymbolicContext,
                x0: &<$t0 as KnownType>::Symbolic,
            ) -> Self::Output {
                let sig = UnarySignature {
                    arg0: <<$t0 as KnownType>::Symbolic as KnownType>::TY,
                    ret: <<$u as KnownType>::Symbolic as KnownType>::TY,
                };
                let op = $op::from_placement_signature(&self, sig);
                ctx.execute(op.into(), vec![x0.clone().into()])
                    .try_into()
                    .unwrap()
            }
        }
    };

    /*
    Binary
    */
    ($t:ident, $plc:ty, ($t0:ty, $t1:ty) -> $u:ty, $op:ident) => {
        impl BinaryKernelCheck<ConcreteContext, $plc, $t0, $t1, $u> for $op {
            fn check(ctx: &ConcreteContext, plc: &$plc, x0: $t0, x1: $t1) -> $u {
                <Self as BinaryKernel<ConcreteContext, $plc, $t0, $t1, $u>>::kernel(
                    ctx, plc, x0, x1,
                )
            }
        }

        impl $t<ConcreteContext, $t0, $t1> for $plc {
            type Output = $u;

            fn apply(&self, ctx: &ConcreteContext, x0: &$t0, x1: &$t1) -> Self::Output {
                let sig = BinarySignature {
                    arg0: <$t0 as KnownType>::TY,
                    arg1: <$t1 as KnownType>::TY,
                    ret: <$u as KnownType>::TY,
                };
                let op = $op::from_placement_signature(&self, sig);
                ctx.execute(op.into(), vec![x0.clone().into(), x1.clone().into()])
                    .try_into()
                    .unwrap()
            }
        }

        impl $t<SymbolicContext, <$t0 as KnownType>::Symbolic, <$t1 as KnownType>::Symbolic>
            for $plc
        {
            type Output = <$u as KnownType>::Symbolic;

            fn apply(
                &self,
                ctx: &SymbolicContext,
                x0: &<$t0 as KnownType>::Symbolic,
                x1: &<$t1 as KnownType>::Symbolic,
            ) -> Self::Output {
                let sig = BinarySignature {
                    arg0: <<$t0 as KnownType>::Symbolic as KnownType>::TY,
                    arg1: <<$t1 as KnownType>::Symbolic as KnownType>::TY,
                    ret: <<$u as KnownType>::Symbolic as KnownType>::TY,
                };
                let op = $op::from_placement_signature(&self, sig);
                ctx.execute(op.into(), vec![x0.clone().into(), x1.clone().into()])
                    .try_into()
                    .unwrap()
            }
        }
    };

    /*
    Ternary
    */
    ($t:ident, $plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty, $op:ident) => {
        impl TernaryKernelCheck<ConcreteContext, $plc, $t0, $t1, $t2, $u> for $op {
            fn check(ctx: &ConcreteContext, plc: &$plc, x0: $t0, x1: $t1, x2: $t2) -> $u {
                <Self as TernaryKernel<ConcreteContext, $plc, $t0, $t1, $t2, $u>>::kernel(
                    ctx, plc, x0, x1, x2,
                )
            }
        }

        impl $t<ConcreteContext, $t0, $t1, $t2> for $plc {
            type Output = $u;

            fn apply(&self, ctx: &ConcreteContext, x0: &$t0, x1: &$t1, x2: &$t2) -> Self::Output {
                let sig = TernarySignature {
                    arg0: <$t0 as KnownType>::TY,
                    arg1: <$t1 as KnownType>::TY,
                    arg2: <$t2 as KnownType>::TY,
                    ret: <$u as KnownType>::TY,
                };
                let op = $op::from_placement_signature(&self, sig);
                ctx.execute(
                    op.into(),
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

            fn apply(
                &self,
                ctx: &SymbolicContext,
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
                let op = $op::from_placement_signature(&self, sig);
                ctx.execute(
                    op.into(),
                    vec![x0.clone().into(), x1.clone().into(), x2.clone().into()],
                )
                .try_into()
                .unwrap()
            }
        }
    };
}

trait PlacementAdd<C: Context, T, U> {
    type Output;

    fn apply(&self, ctx: &C, x: &T, y: &U) -> Self::Output;

    fn add(&self, ctx: &C, x: &T, y: &U) -> Self::Output {
        self.apply(ctx, x, y)
    }
}

trait PlacementSub<C: Context, T, U> {
    type Output;

    fn apply(&self, ctx: &C, x: &T, y: &U) -> Self::Output;

    fn sub(&self, ctx: &C, x: &T, y: &U) -> Self::Output {
        self.apply(ctx, x, y)
    }
}

trait PlacementMul<C: Context, T, U> {
    type Output;

    fn apply(&self, ctx: &C, x: &T, y: &U) -> Self::Output;

    fn mul(&self, ctx: &C, x: &T, y: &U) -> Self::Output {
        self.apply(ctx, x, y)
    }
}

trait PlacementXor<C: Context, T, U> {
    type Output;

    fn apply(&self, ctx: &C, x: &T, y: &U) -> Self::Output;

    fn xor(&self, ctx: &C, x: &T, y: &U) -> Self::Output {
        self.apply(ctx, x, y)
    }
}

trait PlacementAnd<C: Context, T, U> {
    type Output;

    fn apply(&self, ctx: &C, x: &T, y: &U) -> Self::Output;

    fn and(&self, ctx: &C, x: &T, y: &U) -> Self::Output {
        self.apply(ctx, x, y)
    }
}

trait PlacementMulSetup<C: Context, S, T, U> {
    type Output;

    fn apply(&self, ctx: &C, s: &S, x: &T, y: &U) -> Self::Output;

    fn mul(&self, ctx: &C, s: &S, x: &T, y: &U) -> Self::Output {
        self.apply(ctx, s, x, y)
    }
}

trait PlacementShare<C: Context, T> {
    type Output;

    fn apply(&self, ctx: &C, x: &T) -> Self::Output;

    fn share(&self, ctx: &C, x: &T) -> Self::Output {
        self.apply(ctx, x)
    }
}

pub trait Context {
    type Value;
    fn execute(&self, op: Operator, operands: Vec<Self::Value>) -> Self::Value;
}

#[derive(Clone, Debug, Default)]
pub struct ConcreteContext {}

impl Context for ConcreteContext {
    type Value = Value;

    fn execute(&self, op: Operator, operands: Vec<Value>) -> Value {
        match op {
            Operator::PrfKeyGenOp(op) => op.compile(self)(operands),
            Operator::RingSampleOp(op) => op.compile(self)(operands),
            Operator::BitSampleOp(op) => op.compile(self)(operands),
            Operator::RingAddOp(op) => op.compile(self)(operands),
            Operator::BitXorOp(op) => op.compile(self)(operands),
            Operator::BitAndOp(op) => op.compile(self)(operands),
            Operator::RingSubOp(op) => op.compile(self)(operands),
            Operator::RingMulOp(op) => op.compile(self)(operands),
            Operator::RepSetupOp(op) => op.compile(self)(operands),
            Operator::RepShareOp(op) => op.compile(self)(operands),
            Operator::RepRevealOp(op) => op.compile(self)(operands),
            Operator::RepAddOp(op) => op.compile(self)(operands),
            Operator::RepMulOp(op) => op.compile(self)(operands),
            Operator::ConstantOp(op) => op.compile(self)(operands),
        }
    }
}

use std::sync::{Arc, RwLock};

#[derive(Clone, Debug, Default)]
pub struct SymbolicContext {
    ops: Arc<RwLock<Vec<Operation>>>, // TODO use HashMap so we can do some consistency checks on the fly?
}

impl Context for SymbolicContext {
    type Value = SymbolicValue;

    fn execute(&self, op: Operator, operands: Vec<SymbolicValue>) -> SymbolicValue {
        match op {
            Operator::PrfKeyGenOp(op) => op.execute_symbolic(self, operands),
            Operator::RingSampleOp(op) => op.execute_symbolic(self, operands),
            Operator::BitSampleOp(op) => op.execute_symbolic(self, operands),
            Operator::RingAddOp(op) => op.execute_symbolic(self, operands),
            Operator::BitXorOp(op) => op.execute_symbolic(self, operands),
            Operator::BitAndOp(op) => op.execute_symbolic(self, operands),
            Operator::RingSubOp(op) => op.execute_symbolic(self, operands),
            Operator::RingMulOp(op) => op.execute_symbolic(self, operands),
            Operator::RepSetupOp(op) => op.execute_symbolic(self, operands),
            Operator::RepShareOp(op) => op.execute_symbolic(self, operands),
            Operator::RepRevealOp(op) => op.execute_symbolic(self, operands),
            Operator::RepAddOp(op) => op.execute_symbolic(self, operands),
            Operator::RepMulOp(op) => op.execute_symbolic(self, operands),
            Operator::ConstantOp(op) => op.execute_symbolic(self, operands),
        }
    }
}

impl SymbolicContext {
    pub fn add_operation<'s, O: Into<Operator> + Clone>(
        &'s self,
        operator: &O,
        operands: &[&str],
    ) -> String {
        let mut ops = self.ops.write().unwrap(); // TODO unwrap
        let op_name: String = format!("op_{}", ops.len());
        let op = Operation {
            name: op_name.clone(),
            operator: operator.clone().into(),
            operands: operands.iter().map(|op| op.to_string()).collect(),
        };
        ops.push(op);
        op_name
    }
}

impl<T> From<RingTensor<T>> for Symbolic<RingTensor<T>> {
    fn from(x: RingTensor<T>) -> Symbolic<RingTensor<T>> {
        Symbolic::Concrete(x)
    }
}

impl<R> From<ReplicatedTensor<R>> for Symbolic<ReplicatedTensor<R>> {
    fn from(x: ReplicatedTensor<R>) -> Symbolic<ReplicatedTensor<R>> {
        Symbolic::Concrete(x)
    }
}

impl<K> From<AbstractReplicatedSetup<K>> for Symbolic<AbstractReplicatedSetup<K>> {
    fn from(x: AbstractReplicatedSetup<K>) -> Symbolic<AbstractReplicatedSetup<K>> {
        Symbolic::Concrete(x)
    }
}

macro_rules! runtime_kernel {

    /*
    Nullaray
    */

    ($op:ty, [$(($plc:ty, () -> $u:ty)),+], $k:expr) => {
        $(
        impl NullaryKernel<ConcreteContext, $plc, $u> for $op {
            fn kernel(ctx: &ConcreteContext, plc: &$plc) -> $u {
                $k(ctx, plc)
            }
        }
        )+

        impl $op {
            pub fn compile(&self, ctx: &ConcreteContext) -> Box<dyn Fn(Vec<Value>) -> Value> {
                match (self.plc.ty(), self.sig) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Nullary(NullarySignature{
                                ret: <$u>::TY,
                            })
                        ) => {
                            let plc: $plc = self.plc.clone().try_into().unwrap();
                            let ctx = ctx.clone();
                            Box::new(move |_operands: Vec<Value>| {
                                let y: $u = $k(&ctx, &plc);
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

    ($op:ty, [$(($plc:ty, ($t0:ty) -> $u:ty)),+], $k:expr) => {
        $(
        impl UnaryKernel<ConcreteContext, $plc, $t0, $u> for $op {
            fn kernel(ctx: &ConcreteContext, plc: &$plc, x0: $t0) -> $u {
                $k(ctx, plc, x0)
            }
        }
        )+

        impl $op {
            pub fn compile(&self, ctx: &ConcreteContext) -> Box<dyn Fn(Vec<Value>) -> Value> {
                match (self.plc.ty(), self.sig) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Unary(UnarySignature{
                                arg0: <$t0>::TY,
                                ret: <$u>::TY,
                            })
                        ) => {
                            let plc: $plc = self.plc.clone().try_into().unwrap();
                            let ctx = ctx.clone();
                            Box::new(move |operands: Vec<Value>| {
                                let x0: $t0 = operands.get(0).unwrap().clone().try_into().unwrap();

                                let y: $u = $k(&ctx, &plc, x0);
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

    ($op:ty, [$(($plc:ty, ($t0:ty, $t1:ty) -> $u:ty)),+], $k:expr) => {
        $(
        impl BinaryKernel<ConcreteContext, $plc, $t0, $t1, $u> for $op {
            fn kernel(ctx: &ConcreteContext, plc: &$plc, x0: $t0, x1: $t1) -> $u {
                $k(ctx, plc, x0, x1)
            }
        }
        )+

        impl $op {
            pub fn compile(&self, ctx: &ConcreteContext) -> Box<dyn Fn(Vec<Value>) -> Value> {
                match (self.plc.ty(), self.sig) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Binary(BinarySignature{
                                arg0: <$t0>::TY,
                                arg1: <$t1>::TY,
                                ret: <$u>::TY,
                            })
                        ) => {
                            let plc: $plc = self.plc.clone().try_into().unwrap();
                            let ctx = ctx.clone();
                            let op = self.clone();
                            Box::new(move |operands| -> Value {
                                let x0: $t0 = operands.get(0).unwrap().clone().try_into().unwrap();
                                let x1: $t1 = operands.get(1).unwrap().clone().try_into().unwrap();

                                let y: $u = $k(&ctx, &plc, x0, x1);
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

    ($op:ty, [$(($plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty)),+], $k:expr) => {
        $(
        impl TernaryKernel<ConcreteContext, $plc, $t0, $t1, $t2, $u> for $op {
            fn kernel(ctx: &ConcreteContext, plc: &$plc, x0: $t0, x1: $t1, x2: $t2) -> $u {
                $k(ctx, plc, x0, x1, x2)
            }
        }
        )+

        impl $op {
            pub fn compile(&self, ctx: &ConcreteContext) -> Box<dyn Fn(Vec<Value>) -> Value> {
                match (self.plc.ty(), self.sig) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Ternary(TernarySignature{
                                arg0: <$t0>::TY,
                                arg1: <$t1>::TY,
                                arg2: <$t2>::TY,
                                ret: <$u>::TY,
                            })
                        ) => {
                            let plc: $plc = self.plc.clone().try_into().unwrap();
                            let ctx = ctx.clone();
                            let op = self.clone();
                            Box::new(move |operands: Vec<Value>| -> Value {
                                let x0: $t0 = operands.get(0).unwrap().clone().try_into().unwrap();
                                let x1: $t1 = operands.get(1).unwrap().clone().try_into().unwrap();
                                let x2: $t2 = operands.get(2).unwrap().clone().try_into().unwrap();

                                let y = $k(&ctx, &plc, x0, x1, x2);
                                y.into()
                            })
                        }
                    )+
                    _ => unimplemented!(), // ok
                }
            }
        }
    };

}

macro_rules! compiletime_kernel {

    /*
    Nullary
    */

    ($op:ty, [$(($plc:ty, () -> $u:ty)),+], $k:expr) => {
        impl $op {
            pub fn execute_symbolic(&self, ctx: &SymbolicContext, _operands: Vec<SymbolicValue>) -> SymbolicValue {
                match (self.plc.ty(), self.sig) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Nullary(NullarySignature{
                                ret: <<$u as KnownType>::Symbolic as KnownType>::TY,
                            })
                        ) => {
                            let plc: $plc = self.plc.clone().try_into().unwrap();

                            let k: fn(&Self, &SymbolicContext, $plc) -> <$u as KnownType>::Symbolic = $k;

                            let y: <$u as KnownType>::Symbolic = k(self, ctx, plc);
                            SymbolicValue::from(y)
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

    ($op:ty, [$(($plc:ty, ($t0:ty) -> $u:ty)),+], $k:expr) => {
        impl $op {
            pub fn execute_symbolic(&self, ctx: &SymbolicContext, operands: Vec<SymbolicValue>) -> SymbolicValue {
                match (self.plc.ty(), self.sig) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Unary(UnarySignature{
                                arg0: <<$t0 as KnownType>::Symbolic as KnownType>::TY,
                                ret: <<$u as KnownType>::Symbolic as KnownType>::TY,
                            })
                        ) => {
                            let plc: $plc = self.plc.clone().try_into().unwrap();

                            let x0: <$t0 as KnownType>::Symbolic = operands.get(0).unwrap().clone().try_into().unwrap();

                            let k: fn(
                                &Self,
                                &SymbolicContext,
                                $plc,
                                <$t0 as KnownType>::Symbolic
                            ) -> <$u as KnownType>::Symbolic = $k;

                            let y: <$u as KnownType>::Symbolic = k(self, ctx, plc, x0);
                            SymbolicValue::from(y)
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

    ($op:ty, [$(($plc:ty, ($t0:ty, $t1:ty) -> $u:ty)),+], $k:expr) => {
        impl $op {
            pub fn execute_symbolic(
                &self,
                ctx: &SymbolicContext,
                operands: Vec<SymbolicValue>,
            ) -> SymbolicValue {
                match (self.plc.ty(), self.sig) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Binary(BinarySignature{
                                arg0: <<$t0 as KnownType>::Symbolic as KnownType>::TY,
                                arg1: <<$t1 as KnownType>::Symbolic as KnownType>::TY,
                                ret: <<$u as KnownType>::Symbolic as KnownType>::TY,
                            })
                        ) => {
                            let plc: $plc = self.plc.clone().try_into().unwrap();

                            let x0: <$t0 as KnownType>::Symbolic = operands.get(0).unwrap().clone().try_into().unwrap();
                            let x1: <$t1 as KnownType>::Symbolic = operands.get(1).unwrap().clone().try_into().unwrap();

                            let k: fn(
                                &Self,
                                &SymbolicContext,
                                $plc,
                                <$t0 as KnownType>::Symbolic,
                                <$t1 as KnownType>::Symbolic
                            ) -> <$u as KnownType>::Symbolic = $k;

                            let y: <$u as KnownType>::Symbolic = k(self, ctx, plc, x0, x1);
                            SymbolicValue::from(y)
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

    ($op:ty, [$(($plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty)),+], $k:expr) => {
        impl $op {
            pub fn execute_symbolic(
                &self,
                ctx: &SymbolicContext,
                operands: Vec<SymbolicValue>,
            ) -> SymbolicValue {
                match (self.plc.ty(), self.sig) {
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
                            let plc: $plc = self.plc.clone().try_into().unwrap();

                            let x0: <$t0 as KnownType>::Symbolic = operands.get(0).unwrap().clone().try_into().unwrap();
                            let x1: <$t1 as KnownType>::Symbolic = operands.get(1).unwrap().clone().try_into().unwrap();
                            let x2: <$t2 as KnownType>::Symbolic = operands.get(2).unwrap().clone().try_into().unwrap();

                            let k: fn(
                                &Self,
                                &SymbolicContext,
                                $plc,
                                <$t0 as KnownType>::Symbolic,
                                <$t1 as KnownType>::Symbolic,
                                <$t2 as KnownType>::Symbolic,
                            ) -> <$u as KnownType>::Symbolic = $k;

                            let y: <$u as KnownType>::Symbolic = k(self, ctx, plc, x0, x1, x2);
                            SymbolicValue::from(y)
                        }
                    )+
                    _ => unimplemented!(), // ok
                }
            }

        }
    };
}

/// Kernel function is never used in symbolic contexts
macro_rules! kernel {

    /*
    Nullary
    */

    ($op:ty, [$(($plc:ty, () -> $u:ty)),+], $k:expr) => {
        runtime_kernel!($op, [$(($plc, () -> $u)),+], $k);
        compiletime_kernel!($op, [$(($plc, () -> $u)),+], |op, ctx, _plc| {
            let op_name = ctx.add_operation(op, &[]);
            Symbolic::Symbolic(SymbolicHandle { op: op_name })
        });
    };

    /*
    Unary
    */

    ($op:ty, [$(($plc:ty, ($t0:ty) -> $u:ty)),+], $k:expr) => {
        runtime_kernel!($op, [$(($plc, ($t0) -> $u)),+], $k);
        compiletime_kernel!($op, [$(($plc, ($t0) -> $u)),+], |op, ctx, _plc, x0| {
            let x0_op = match x0 {
                Symbolic::Symbolic(h) => h.op,
                Symbolic::Concrete(_) => unimplemented!(),
            };

            let op_name = ctx.add_operation(op, &[&x0_op]);
            Symbolic::Symbolic(SymbolicHandle { op: op_name })
        });
    };

    /*
    Binary
    */

    ($op:ty, [$(($plc:ty, ($t0:ty, $t1:ty) -> $u:ty)),+], $k:expr) => {
        runtime_kernel!($op, [$(($plc, ($t0, $t1) -> $u)),+], $k);
        compiletime_kernel!($op, [$(($plc, ($t0, $t1) -> $u)),+], |op, ctx, _plc, x0, x1| {
            let x0_op = match x0 {
                Symbolic::Symbolic(h) => h.op,
                Symbolic::Concrete(_) => unimplemented!(),
            };

            let x1_op = match x1 {
                Symbolic::Symbolic(h) => h.op,
                Symbolic::Concrete(_) => unimplemented!(),
            };

            let op_name = ctx.add_operation(op, &[&x0_op, &x1_op]);
            Symbolic::Symbolic(SymbolicHandle { op: op_name })
        });
    };

    /*
    Ternary
    */

    ($op:ty, [$(($plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty)),+], $k:expr) => {
        runtime_kernel!($op, [$(($plc, ($t0, $t1, $t2) -> $u)),+], $k);
        compiletime_kernel!($op, [$(($plc, ($t0, $t1, $t2) -> $u)),+], |op, ctx, _plc, x0, x1, x2| {
            let x0_op = match x0 {
                Symbolic::Symbolic(h) => h.op,
                Symbolic::Concrete(_) => unimplemented!(),
            };

            let x1_op = match x1 {
                Symbolic::Symbolic(h) => h.op,
                Symbolic::Concrete(_) => unimplemented!(),
            };

            let x2_op = match x2 {
                Symbolic::Symbolic(h) => h.op,
                Symbolic::Concrete(_) => unimplemented!(),
            };

            let op_name = ctx.add_operation(op, &[&x0_op, &x1_op, &x2_op]);
            Symbolic::Symbolic(SymbolicHandle { op: op_name })
        });
    };
}

/// Kernel function is used to map Symbolic::Concrete to Into<Symbolic> in symbolic contexts
macro_rules! hybrid_kernel {

    /*
    Nullary
    */

    ($op:ty, [$(($plc:ty, () -> $u:ty)),+], $k:expr) => {
        runtime_kernel!($op, [$(($plc, () -> $u)),+], $k);
        compiletime_kernel!($op, [$(($plc, () -> $u)),+], |_op, ctx, plc| {
            $k(ctx, &plc).into()
        });
    };

    /*
    Unary
    */

    ($op:ty, [$(($plc:ty, ($t0:ty) -> $u:ty)),+], $k:expr) => {
        runtime_kernel!($op, [$(($plc, ($t0) -> $u)),+], $k);
        compiletime_kernel!($op, [$(($plc, ($t0) -> $u)),+], |op, ctx, plc, x0| {
            match x0 {
                Symbolic::Concrete(x0) => {
                    $k(ctx, &plc, x0).into()
                }
                Symbolic::Symbolic(h0) => {
                    let op_name = ctx.add_operation(op, &[&h0.op]);
                    Symbolic::Symbolic(SymbolicHandle { op: op_name })
                }
            }
        });
    };

    /*
    Binary
    */

    ($op:ty, [$(($plc:ty, ($t0:ty, $t1:ty) -> $u:ty)),+], $k:expr) => {
        runtime_kernel!($op, [$(($plc, ($t0, $t1) -> $u)),+], $k);
        compiletime_kernel!($op, [$(($plc, ($t0, $t1) -> $u)),+], |op, ctx, plc, x0, x1| {
            match (x0, x1) {
                (Symbolic::Concrete(x0), Symbolic::Concrete(x1)) => {
                    $k(ctx, &plc, x0, x1).into()
                }
                (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1)) => {
                    let op_name = ctx.add_operation(op, &[&h0.op, &h1.op]);
                    Symbolic::Symbolic(SymbolicHandle { op: op_name })
                }
                _ => unimplemented!(), // ok
            }
        });
    };

    /*
    Ternary
    */

    ($op:ty, [$(($plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty)),+], $k:expr) => {
        runtime_kernel!($op, [$(($plc, ($t0, $t1, $t2) -> $u)),+], $k);
        compiletime_kernel!($op, [$(($plc, ($t0, $t1, $t2) -> $u)),+], |op, ctx, plc, x0, x1, x2| {
            match (x0, x1, x2) {
                (Symbolic::Concrete(x0), Symbolic::Concrete(x1), Symbolic::Concrete(x2)) => {
                    $k(ctx, &plc, x0, x1, x2).into()
                }
                (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1), Symbolic::Symbolic(h2)) => {
                    let op_name = ctx.add_operation(op, &[&h0.op, &h1.op, &h2.op]);
                    Symbolic::Symbolic(SymbolicHandle { op: op_name })
                }
                _ => unimplemented!(), // ok
            }
        });
    };
}

/// Kernel function is used to map Symbolic to Into<Symbolic> in symbolic contexts
macro_rules! abstract_kernel {

    /*
    Nullary
    */

    ($op:ty, [$(($plc:ty, () -> $u:ty)),+], $k:expr) => {
        runtime_kernel!($op, [$(($plc, () -> $u)),+], $k);
        compiletime_kernel!($op, [$(($plc, () -> $u)),+], |_op, ctx, plc| {
            $k(ctx, &plc).into()
        });
    };

    /*
    Unary
    */

    ($op:ty, [$(($plc:ty, ($t0:ty) -> $u:ty)),+], $k:expr) => {
        runtime_kernel!($op, [$(($plc, ($t0) -> $u)),+], $k);
        compiletime_kernel!($op, [$(($plc, ($t0) -> $u)),+], |_op, ctx, plc, x0| {
            $k(ctx, &plc, x0).into()
        });
    };

    /*
    Binary
    */

    ($op:ty, [$(($plc:ty, ($t0:ty, $t1:ty) -> $u:ty)),+], $k:expr) => {
        runtime_kernel!($op, [$(($plc, ($t0, $t1) -> $u)),+], $k);
        compiletime_kernel!($op, [$(($plc, ($t0, $t1) -> $u)),+], |_op, ctx, plc, x0, x1| {
            $k(ctx, &plc, x0, x1).into()
        });
    };

    /*
    Ternary
    */

    ($op:ty, [$(($plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty)),+], $k:expr) => {
        runtime_kernel!($op, [$(($plc, ($t0, $t1, $t2) -> $u)),+], $k);
        compiletime_kernel!($op, [$(($plc, ($t0, $t1, $t2) -> $u)),+], |_op, ctx, plc, x0, x1, x2| {
            $k(ctx, &plc, x0, x1, x2).into()
        });
    };
}

pub trait NullaryKernel<C: Context, P, Y> {
    fn kernel(ctx: &C, plc: &P) -> Y;
}

pub trait UnaryKernel<C: Context, P, X0, Y> {
    fn kernel(ctx: &C, plc: &P, x0: X0) -> Y;
}

pub trait BinaryKernel<C: Context, P, X0, X1, Y> {
    fn kernel(ctx: &C, plc: &P, x0: X0, x1: X1) -> Y;
}

pub trait TernaryKernel<C: Context, P, X0, X1, X2, Y> {
    fn kernel(ctx: &C, plc: &P, x0: X0, x1: X1, x2: X2) -> Y;
}

trait NullaryKernelCheck<C: Context, P, Y> {
    fn check(ctx: &C, plc: &P) -> Y;
}

trait UnaryKernelCheck<C: Context, P, X0, Y> {
    fn check(ctx: &C, plc: &P, x0: X0) -> Y;
}

trait BinaryKernelCheck<C: Context, P, X0, X1, Y> {
    fn check(ctx: &C, plc: &P, x0: X0, x1: X1) -> Y;
}

trait TernaryKernelCheck<C: Context, P, X0, X1, X2, Y> {
    fn check(ctx: &C, plc: &P, x0: X0, x1: X1, x2: X2) -> Y;
}

macro_rules! apply {
    ($plc:ident, $ctx:ident, $f:expr, $($x:expr),+) => {
        {
            let y = $f(
                $(
                    &$plc.place($ctx, $x)
                ),+
            );
            y.val.unwrap()
        }
    };
}

enum PlacedValue<'v, V: Clone> {
    Borrowed(&'v V),
    Owned(V),
}

impl<'v, V: Clone> PlacedValue<'v, V> {
    // TODO use ToOwned trait?
    fn unwrap(self) -> V {
        match self {
            PlacedValue::Borrowed(r) => r.clone(),
            PlacedValue::Owned(v) => v,
        }
    }

    // TODO use AsRef trait?
    // TODO loosen lifetime bounds?
    fn as_ref(&'v self) -> &'v V {
        match self {
            PlacedValue::Borrowed(r) => *r,
            PlacedValue::Owned(v) => v,
        }
    }
}

struct Placed<'v, 'p, 'c, V: Clone, P, C> {
    val: PlacedValue<'v, V>,
    plc: &'p P,
    ctx: &'c C,
}

impl HostPlacement {
    fn place<'v, 'p, 'c, C: Context, V: Clone>(
        &'p self,
        ctx: &'c C,
        val: &'v V,
    ) -> Placed<'v, 'p, 'c, V, HostPlacement, C> {
        Placed {
            val: PlacedValue::Borrowed(val),
            plc: self,
            ctx,
        }
    }
}

macro_rules! placed_op_impl {
    ($t:ident::$f:ident, $pt:ident) => {
        impl<'x, 'y, 'p, 'c, V, P, C: Context> $t<Placed<'y, 'p, 'c, V, P, C>>
            for Placed<'x, 'p, 'c, V, P, C>
        where
            V: Clone,
            V: 'static,
            P: PartialEq + std::fmt::Debug,
            P: $pt<C, V, V, Output = V>,
        {
            type Output = Placed<'static, 'p, 'c, V, P, C>;

            fn $f(self, other: Placed<'y, 'p, 'c, V, P, C>) -> Self::Output {
                $t::$f(&self, &other)
            }
        }

        impl<'x, 'y, 'p, 'c, V, P, C: Context> $t<&Placed<'y, 'p, 'c, V, P, C>>
            for Placed<'x, 'p, 'c, V, P, C>
        where
            V: Clone,
            V: 'static,
            P: PartialEq + std::fmt::Debug,
            P: $pt<C, V, V, Output = V>,
        {
            type Output = Placed<'static, 'p, 'c, V, P, C>;

            fn $f(self, other: &Placed<'y, 'p, 'c, V, P, C>) -> Self::Output {
                $t::$f(&self, other)
            }
        }

        impl<'x, 'y, 'p, 'c, V, P, C: Context> $t<Placed<'y, 'p, 'c, V, P, C>>
            for &Placed<'x, 'p, 'c, V, P, C>
        where
            V: Clone,
            V: 'static,
            P: PartialEq + std::fmt::Debug,
            P: $pt<C, V, V, Output = V>,
        {
            type Output = Placed<'static, 'p, 'c, V, P, C>;

            fn $f(self, other: Placed<'y, 'p, 'c, V, P, C>) -> Self::Output {
                $t::$f(self, &other)
            }
        }

        impl<'x, 'y, 'p, 'c, V, P, C: Context> $t<&Placed<'y, 'p, 'c, V, P, C>>
            for &Placed<'x, 'p, 'c, V, P, C>
        where
            V: Clone,
            V: 'static,
            P: PartialEq + std::fmt::Debug,
            P: $pt<C, V, V, Output = V>,
        {
            type Output = Placed<'static, 'p, 'c, V, P, C>;

            fn $f(self, other: &Placed<'y, 'p, 'c, V, P, C>) -> Self::Output {
                let Placed {
                    val: x,
                    plc: x_plc,
                    ctx: x_ctx,
                } = self;
                let Placed {
                    val: y,
                    plc: y_plc,
                    ctx: y_ctx,
                } = other;
                assert_eq!(x_plc, y_plc); // TODO if we do this properly we could get rid of this check
                let plc = *x_plc;
                let ctx = *x_ctx;

                let y = plc.$f(ctx, x.as_ref(), y.as_ref());
                Placed {
                    val: PlacedValue::Owned(y),
                    plc,
                    ctx,
                }
            }
        }
    };
}

placed_op_impl!(Add::add, PlacementAdd);
placed_op_impl!(Sub::sub, PlacementSub);
placed_op_impl!(Mul::mul, PlacementMul);

#[derive(Clone, Debug, PartialEq)]
pub struct RepSetupOp {
    sig: Signature,
    plc: Placement,
}

impl RepSetupOp {
    fn kernel<C: Context, K: Clone>(
        ctx: &C,
        rep: &ReplicatedPlacement,
    ) -> AbstractReplicatedSetup<K>
    where
        HostPlacement: PlacementKeyGen<C, K>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let k0 = player0.keygen(ctx);
        let k1 = player1.keygen(ctx);
        let k2 = player2.keygen(ctx);

        AbstractReplicatedSetup {
            keys: [[k0.clone(), k1.clone()], [k1, k2.clone()], [k2, k0]],
        }
    }
}

hybrid_kernel! {
    RepSetupOp,
    [
        (ReplicatedPlacement, () -> ReplicatedSetup)
    ],
    Self::kernel
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepAddOp {
    sig: Signature,
    plc: Placement,
}

macro_rules! with_player {
    // Collection of assignment blocks
    ($player:ident, $context:ident { $( $var:tt = [$($right:tt)*])* }  ) => {
        $( let $var = eval_with_context!($player, $context, $($right)*); )*
    };
}

impl RepAddOp {
    fn from_placement_signature(plc: &ReplicatedPlacement, sig: BinarySignature) -> Self {
        RepAddOp {
            sig: sig.into(),
            plc: plc.clone().into(),
        }
    }

    fn kernel<C: Context, R>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        x: ReplicatedTensor<R>,
        y: ReplicatedTensor<R>,
    ) -> ReplicatedTensor<R>
    where
        R: Clone,
        HostPlacement: PlacementAdd<C, R, R, Output = R>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let ReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let ReplicatedTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        with_player!( player0, ctx {
            z00 = [x00 + y00]
            z10 = [x10 + y10]
        });

        with_player!( player1, ctx {
            z11 = [x11 + y11]
            z21 = [x21 + y21]
        });

        with_player!( player2, ctx {
            z22 = [x22 + y22]
            z02 = [x02 + y02]
        });

        /* Produces code identical to this one:
        let z00 = player0.add(ctx, x00, y00);
        let z10 = player0.add(ctx, x10, y10);

        let z11 = player1.add(ctx, x11, y11);
        let z21 = player1.add(ctx, x21, y21);

        let z22 = player2.add(ctx, x22, y22);
        let z02 = player2.add(ctx, x02, y02);
        */

        ReplicatedTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
    }
}

modelled!(PlacementAdd, ReplicatedPlacement, (Replicated64Tensor, Replicated64Tensor) -> Replicated64Tensor, RepAddOp);
modelled!(PlacementAdd, ReplicatedPlacement, (Replicated128Tensor, Replicated128Tensor) -> Replicated128Tensor, RepAddOp);
modelled!(PlacementAdd, ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor, RepAddOp);

hybrid_kernel! {
    RepAddOp,
    [
        (ReplicatedPlacement, (Replicated64Tensor, Replicated64Tensor) -> Replicated64Tensor),
        (ReplicatedPlacement, (Replicated128Tensor, Replicated128Tensor) -> Replicated128Tensor),
        (ReplicatedPlacement, (ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor)
    ],
    Self::kernel
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepMulOp {
    sig: Signature,
    plc: Placement,
}

impl RepMulOp {
    fn from_placement_signature(plc: &ReplicatedPlacement, sig: TernarySignature) -> Self {
        RepMulOp {
            sig: sig.into(),
            plc: plc.clone().into(),
        }
    }

    fn kernel<C: Context, R, K>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        s: AbstractReplicatedSetup<K>,
        x: ReplicatedTensor<R>,
        y: ReplicatedTensor<R>,
    ) -> ReplicatedTensor<R>
    where
        R: Clone + Into<C::Value> + TryFrom<C::Value> + 'static,
        HostPlacement: PlacementSample<C, R>,
        HostPlacement: PlacementAdd<C, R, R, Output = R>,
        HostPlacement: PlacementMul<C, R, R, Output = R>,
        ReplicatedPlacement: PlacementZeroShare<C, K, R>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let ReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let ReplicatedTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        let ReplicatedZeroShare {
            alphas: [a0, a1, a2],
        } = rep.zero_share(ctx, &s);

        // TODO because of Rust's let polymorphism and lifetimes we cannot use the same f in all apply! below
        // let f = |xii, yii, xji, yji| {
        //     xii * yii + xii * yji + xji * yii
        // };
        // let z0 = apply!(
        //     player0,
        //     ctx,
        //     |xii, yii, xji, yji, ai| { xii * yii + xii * yji + xji * yii + ai },
        //     x00,
        //     y00,
        //     x10,
        //     y10,
        //     &a0
        // );
        // let z1 = apply!(
        //     player1,
        //     ctx,
        //     |xii, yii, xji, yji, ai| { xii * yii + xii * yji + xji * yii + ai },
        //     x11,
        //     y11,
        //     x21,
        //     y21,
        //     &a1
        // );
        // let z2 = apply!(
        //     player2,
        //     ctx,
        //     |xii, yii, xji, yji, ai| { xii * yii + xii * yji + xji * yii + ai },
        //     x22,
        //     y22,
        //     x02,
        //     y02,
        //     &a2
        // );

        with_player!(player0, ctx {
          z0 = [x00 * y00 + x00 * y10 + x10 * y00 + a0]
        });
        with_player!(player1, ctx {
          z1 = [x11 * y11 + x11 * y21 + x21 * y11 + a1]
        });
        with_player!(player2, ctx {
          z2 = [x22 * y22 + x22 * y02 + x02 * y22 + a2]
        });

        ReplicatedTensor {
            shares: [[z0.clone(), z1.clone()], [z1, z2.clone()], [z2, z0]],
        }
    }
}

modelled!(PlacementMulSetup, ReplicatedPlacement, (ReplicatedSetup, Replicated64Tensor, Replicated64Tensor) -> Replicated64Tensor, RepMulOp);
modelled!(PlacementMulSetup, ReplicatedPlacement, (ReplicatedSetup, Replicated128Tensor, Replicated128Tensor) -> Replicated128Tensor, RepMulOp);
modelled!(PlacementMulSetup, ReplicatedPlacement, (ReplicatedSetup, ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor, RepMulOp);

hybrid_kernel! {
    RepMulOp,
    [
        (ReplicatedPlacement, (ReplicatedSetup, Replicated64Tensor, Replicated64Tensor) -> Replicated64Tensor),
        (ReplicatedPlacement, (ReplicatedSetup, Replicated128Tensor, Replicated128Tensor) -> Replicated128Tensor),
        (ReplicatedPlacement, (ReplicatedSetup, ReplicatedBitTensor, ReplicatedBitTensor) -> ReplicatedBitTensor)
    ],
    Self::kernel
}

trait PlacementZeroShare<C: Context, K, R> {
    fn zero_share(&self, ctx: &C, setup: &AbstractReplicatedSetup<K>) -> ReplicatedZeroShare<R>;
}

// NOTE this is an un-modelled operation (as opposed to the modelled! operations that have
// a representation in computations); should we have a macro for this as well?
impl<C: Context, K, R> PlacementZeroShare<C, K, R> for ReplicatedPlacement
where
    R: Clone + 'static,
    HostPlacement: PlacementSample<C, R>,
    HostPlacement: PlacementSub<C, R, R, Output = R>,
{
    fn zero_share(&self, ctx: &C, s: &AbstractReplicatedSetup<K>) -> ReplicatedZeroShare<R> {
        let (player0, player1, player2) = self.host_placements();

        let AbstractReplicatedSetup {
            keys: [[k00, k10], [k11, k21], [k22, k02]],
        } = s;

        // TODO use keys when sampling!

        let r00 = player0.sample(ctx);
        let r10 = player0.sample(ctx);
        // let alpha1 = player0.sub(ctx, &r00, &r10);
        let alpha0 = apply!(player0, ctx, |x, y| { x - y }, &r00, &r10);

        let r11 = player1.sample(ctx);
        let r21 = player1.sample(ctx);
        // let alpha1 = player1.sub(ctx, &r11, &r21);
        let alpha1 = apply!(player1, ctx, |x, y| { x - y }, &r11, &r21);

        let r22 = player2.sample(ctx);
        let r02 = player2.sample(ctx);
        // let alpha2 = player2.sub(ctx, &r22, &r02);
        let alpha2 = apply!(player2, ctx, |x, y| { x - y }, &r22, &r02);

        ReplicatedZeroShare {
            alphas: [alpha0, alpha1, alpha2],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepShareOp {
    sig: Signature,
    plc: Placement,
}

impl RepShareOp {
    fn from_placement_signature(plc: &ReplicatedPlacement, sig: UnarySignature) -> Self {
        RepShareOp {
            sig: sig.into(),
            plc: plc.clone().into(),
        }
    }

    fn kernel<C: Context, R: Clone>(ctx: &C, rep: &ReplicatedPlacement, x: R) -> ReplicatedTensor<R>
    where
        R: Into<C::Value> + TryFrom<C::Value> + 'static,
        HostPlacement: PlacementSample<C, R>,
        HostPlacement: PlacementAdd<C, R, R, Output = R>,
        HostPlacement: PlacementSub<C, R, R, Output = R>,
    {
        let (player0, player1, player2) = rep.host_placements();

        // TODO we should not use player0 here, but rather the placement of `x` (which is currently not implemented)
        let x0 = player0.sample(ctx);
        let x1 = player0.sample(ctx);
        let x2 = apply!(player0, ctx, |x, x0, x1| { x - (x0 + x1) }, &x, &x0, &x1);

        ReplicatedTensor {
            shares: [[x0.clone(), x1.clone()], [x1, x2.clone()], [x2, x0]],
        }
    }
}

modelled!(PlacementShare, ReplicatedPlacement, (Ring64Tensor) -> Replicated64Tensor, RepShareOp);
modelled!(PlacementShare, ReplicatedPlacement, (Ring128Tensor) -> Replicated128Tensor, RepShareOp);
modelled!(PlacementShare, ReplicatedPlacement, (BitTensor) -> ReplicatedBitTensor, RepShareOp);

abstract_kernel! {
    RepShareOp,
    [
        (ReplicatedPlacement, (Ring64Tensor) -> Replicated64Tensor),
        (ReplicatedPlacement, (Ring128Tensor) -> Replicated128Tensor),
        (ReplicatedPlacement, (BitTensor) -> ReplicatedBitTensor)
    ],
    Self::kernel
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepRevealOp {
    sig: Signature,
    plc: Placement,
}

hybrid_kernel! {
    RepRevealOp,
    [
        (ReplicatedPlacement, (Replicated64Tensor) -> Ring64Tensor),
        (ReplicatedPlacement, (Replicated128Tensor) -> Ring128Tensor)
    ],
    Self::kernel
}

impl RepRevealOp {
    fn from_placement_signature(plc: &ReplicatedPlacement, sig: UnarySignature) -> Self {
        RepRevealOp {
            sig: sig.into(),
            plc: plc.clone().into(),
        }
    }

    fn kernel<C: Context, R: Clone>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        xe: ReplicatedTensor<R>,
    ) -> R
    where
        R: Clone + 'static,
        HostPlacement: PlacementAdd<C, R, R, Output = R>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let ReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &xe;

        // TODO we should not use player0 here, but rather the placement of `x` (which is currently not implemented)
        // player0.add(ctx, &player0.add(ctx, x00, x10), x21)
        apply!(player0, ctx, |x0, x1, x2| { x0 + x1 + x2 }, x00, x10, x21)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RingAddOp {
    sig: Signature,
    plc: Placement, // TODO placement should be on Operation!
}

impl RingAddOp {
    fn from_placement_signature(plc: &HostPlacement, sig: BinarySignature) -> Self {
        RingAddOp {
            sig: sig.into(),
            plc: plc.clone().into(),
        }
    }

    fn kernel<C: Context, T>(
        _ctx: &C,
        _plc: &HostPlacement,
        x: RingTensor<T>,
        y: RingTensor<T>,
    ) -> RingTensor<T>
    where
        RingTensor<T>: Add<RingTensor<T>, Output = RingTensor<T>>,
    {
        x + y
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BitXorOp {
    sig: Signature,
    plc: Placement, // TODO placement should be on Operation!
}

impl BitXorOp {
    fn from_placement_signature(plc: &HostPlacement, sig: BinarySignature) -> Self {
        BitXorOp {
            sig: sig.into(),
            plc: plc.clone().into(),
        }
    }

    fn kernel<C: Context>(_ctx: &C, _plc: &HostPlacement, x: BitTensor, y: BitTensor) -> BitTensor
    where
        BitTensor: BitXor<BitTensor, Output = BitTensor>,
    {
        x ^ y
    }
}

// NOTE uncomment the next line to see the kernel check system in action
// modelled!(PlacementAdd, HostPlacement, (Ring32Tensor, Ring32Tensor) -> Ring32Tensor, RingAddOp);
// NOTE that supporting op attributes might be a simple adding an ctor input to the macro: (Placement, Signature) -> Op
modelled!(PlacementAdd, HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor, RingAddOp);
modelled!(PlacementAdd, HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor, RingAddOp);
modelled!(PlacementXor, HostPlacement, (BitTensor, BitTensor) -> BitTensor, BitXorOp);

impl PlacementAdd<ConcreteContext, BitTensor, BitTensor> for HostPlacement {
    type Output = BitTensor;
    fn apply(&self, ctx: &ConcreteContext, x: &BitTensor, y: &BitTensor) -> BitTensor {
        // NOTE: xor = add when in Z2
        self.xor(ctx, x, y)
    }
}

impl
    PlacementAdd<
        SymbolicContext,
        <BitTensor as KnownType>::Symbolic,
        <BitTensor as KnownType>::Symbolic,
    > for HostPlacement
{
    type Output = <BitTensor as KnownType>::Symbolic;
    fn apply(
        &self,
        ctx: &SymbolicContext,
        x: &<BitTensor as KnownType>::Symbolic,
        y: &<BitTensor as KnownType>::Symbolic,
    ) -> Self::Output {
        // NOTE: xor = add when in Z2
        self.xor(ctx, x, y)
    }
}

kernel! {
    RingAddOp,
    [
        (HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor),
        (HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor)
    ],
    Self::kernel
}

kernel! {
    BitXorOp,
    [
        (HostPlacement, (BitTensor, BitTensor) -> BitTensor)
    ],
    Self::kernel
}

#[derive(Clone, Debug, PartialEq)]
pub struct RingSubOp {
    sig: Signature,
    plc: Placement,
}

impl RingSubOp {
    fn from_placement_signature(plc: &HostPlacement, sig: BinarySignature) -> Self {
        RingSubOp {
            sig: sig.into(),
            plc: plc.clone().into(),
        }
    }

    fn kernel<C: Context, T>(
        _ctx: &C,
        _plc: &HostPlacement,
        x: RingTensor<T>,
        y: RingTensor<T>,
    ) -> RingTensor<T>
    where
        RingTensor<T>: Sub<RingTensor<T>, Output = RingTensor<T>>,
    {
        x - y
    }
}

modelled!(PlacementSub, HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor, RingSubOp);
modelled!(PlacementSub, HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor, RingSubOp);

kernel! {
    RingSubOp,
    [
        (HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor),
        (HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor)
    ],
    Self::kernel
}

impl PlacementSub<ConcreteContext, BitTensor, BitTensor> for HostPlacement {
    type Output = BitTensor;
    fn apply(&self, ctx: &ConcreteContext, x: &BitTensor, y: &BitTensor) -> BitTensor {
        // NOTE: xor = sub when in Z2
        self.xor(ctx, x, y)
    }
}

impl
    PlacementSub<
        SymbolicContext,
        <BitTensor as KnownType>::Symbolic,
        <BitTensor as KnownType>::Symbolic,
    > for HostPlacement
{
    type Output = <BitTensor as KnownType>::Symbolic;
    fn apply(
        &self,
        ctx: &SymbolicContext,
        x: &<BitTensor as KnownType>::Symbolic,
        y: &<BitTensor as KnownType>::Symbolic,
    ) -> Self::Output {
        // NOTE: xor = sub when in Z2
        self.xor(ctx, x, y)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RingMulOp {
    sig: Signature,
    plc: Placement,
}

impl RingMulOp {
    fn from_placement_signature(plc: &HostPlacement, sig: BinarySignature) -> Self {
        RingMulOp {
            sig: sig.into(),
            plc: plc.clone().into(),
        }
    }

    fn kernel<C: Context, T>(
        _ctx: &C,
        _plc: &HostPlacement,
        x: RingTensor<T>,
        y: RingTensor<T>,
    ) -> RingTensor<T>
    where
        RingTensor<T>: Mul<RingTensor<T>, Output = RingTensor<T>>,
    {
        x * y
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BitAndOp {
    sig: Signature,
    plc: Placement, // TODO placement should be on Operation!
}

impl BitAndOp {
    fn from_placement_signature(plc: &HostPlacement, sig: BinarySignature) -> Self {
        BitAndOp {
            sig: sig.into(),
            plc: plc.clone().into(),
        }
    }

    fn kernel<C: Context>(_ctx: &C, _plc: &HostPlacement, x: BitTensor, y: BitTensor) -> BitTensor
    where
        BitTensor: BitAnd<BitTensor, Output = BitTensor>,
    {
        x & y
    }
}

modelled!(PlacementMul, HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor, RingMulOp);
modelled!(PlacementMul, HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor, RingMulOp);
modelled!(PlacementAnd, HostPlacement, (BitTensor, BitTensor) -> BitTensor, BitAndOp);

kernel! {
    RingMulOp,
    [
        (HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor),
        (HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor)
    ],
    Self::kernel
}

kernel! {
    BitAndOp,
    [
        (HostPlacement, (BitTensor, BitTensor) -> BitTensor)
    ],
    Self::kernel
}

impl PlacementMul<ConcreteContext, BitTensor, BitTensor> for HostPlacement {
    type Output = BitTensor;
    fn apply(&self, ctx: &ConcreteContext, x: &BitTensor, y: &BitTensor) -> BitTensor {
        // NOTE: mul = and when in Z2
        self.and(ctx, x, y)
    }
}

impl
    PlacementMul<
        SymbolicContext,
        <BitTensor as KnownType>::Symbolic,
        <BitTensor as KnownType>::Symbolic,
    > for HostPlacement
{
    type Output = <BitTensor as KnownType>::Symbolic;
    fn apply(
        &self,
        ctx: &SymbolicContext,
        x: &<BitTensor as KnownType>::Symbolic,
        y: &<BitTensor as KnownType>::Symbolic,
    ) -> Self::Output {
        // NOTE: mul = and when in Z2
        self.and(ctx, x, y)
    }
}

trait PlacementKeyGen<C: Context, K> {
    fn apply(&self, ctx: &C) -> K;

    fn keygen(&self, ctx: &C) -> K {
        self.apply(ctx)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PrfKeyGenOp {
    sig: Signature,
    plc: Placement,
}

modelled!(PlacementKeyGen, HostPlacement, () -> PrfKey, PrfKeyGenOp);

kernel! {
    PrfKeyGenOp,
    [
        (HostPlacement, () -> PrfKey)
    ],
    Self::kernel
}

impl PrfKeyGenOp {
    fn from_placement_signature(plc: &HostPlacement, sig: NullarySignature) -> Self {
        PrfKeyGenOp {
            sig: sig.into(),
            plc: plc.clone().into(),
        }
    }

    fn kernel(ctx: &ConcreteContext, plc: &HostPlacement) -> PrfKey {
        // TODO
        PrfKey([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    }
}

trait PlacementSample<C: Context, O> {
    fn apply(&self, ctx: &C) -> O;

    fn sample(&self, ctx: &C) -> O {
        self.apply(ctx)
    }
}

modelled!(PlacementSample, HostPlacement, () -> Ring64Tensor, RingSampleOp);
modelled!(PlacementSample, HostPlacement, () -> Ring128Tensor, RingSampleOp);
modelled!(PlacementSample, HostPlacement, () -> BitTensor, BitSampleOp);

kernel! {
    RingSampleOp,
    [
        (HostPlacement, () -> Ring64Tensor),
        (HostPlacement, () -> Ring128Tensor)
    ],
    Self::kernel
}

kernel! {
    BitSampleOp,
    [
        (HostPlacement, () -> BitTensor)
    ],
    Self::kernel
}

#[derive(Clone, Debug, PartialEq)]
pub struct RingSampleOp {
    sig: Signature,
    plc: Placement,
}

impl RingSampleOp {
    fn from_placement_signature(plc: &HostPlacement, sig: NullarySignature) -> Self {
        RingSampleOp {
            sig: sig.into(),
            plc: plc.clone().into(),
        }
    }

    fn kernel<T>(ctx: &ConcreteContext, plc: &HostPlacement) -> RingTensor<T>
    where
        T: From<u32>,
    {
        // TODO
        RingTensor::<T>(T::from(987654321))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BitSampleOp {
    sig: Signature,
    plc: Placement,
}

impl BitSampleOp {
    fn from_placement_signature(plc: &HostPlacement, sig: NullarySignature) -> Self {
        BitSampleOp {
            sig: sig.into(),
            plc: plc.clone().into(),
        }
    }

    fn kernel(ctx: &ConcreteContext, plc: &HostPlacement) -> BitTensor
where {
        // TODO
        BitTensor(0)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConstantOp {
    sig: Signature,
    plc: Placement,
    val: Value,
}

impl ConstantOp {
    pub fn compile(&self, _ctx: &ConcreteContext) -> Box<dyn Fn(Vec<Value>) -> Value> {
        let val = self.val.clone();

        match &self.plc {
            Placement::HostPlacement(_) => Box::new(move |_operands| -> Value { val.clone() }),
            _ => unimplemented!(), // ok
        }
    }

    pub fn execute_symbolic(
        &self,
        ctx: &SymbolicContext,
        _operands: Vec<SymbolicValue>,
    ) -> SymbolicValue {
        match &self.plc {
            Placement::HostPlacement(_) => {
                let op_name = ctx.add_operation(self, &[]);
                self.val.ty().synthesize_symbolic_value(op_name)
            }
            _ => unimplemented!(), // ok
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rep_add_concrete() {
        let ctx = ConcreteContext::default();

        let rep = ReplicatedPlacement {
            players: ["alice".into(), "bob".into(), "carole".into()],
        };

        let xe: Replicated64Tensor = ReplicatedTensor {
            shares: [
                [RingTensor(1), RingTensor(2)],
                [RingTensor(2), RingTensor(3)],
                [RingTensor(3), RingTensor(1)],
            ],
        };

        let ye = ReplicatedTensor {
            shares: [
                [RingTensor(1), RingTensor(2)],
                [RingTensor(2), RingTensor(3)],
                [RingTensor(3), RingTensor(1)],
            ],
        };

        let ze: ReplicatedTensor<_> = rep.add(&ctx, &xe, &ye);

        assert_eq!(
            ze,
            ReplicatedTensor {
                shares: [
                    [RingTensor(2), RingTensor(4)],
                    [RingTensor(4), RingTensor(6)],
                    [RingTensor(6), RingTensor(2)],
                ],
            }
        );
    }

    #[test]
    fn test_rep_add_symbolic() {
        let ctx = SymbolicContext::default();

        let rep = ReplicatedPlacement {
            players: ["alice".into(), "bob".into(), "carole".into()],
        };

        let xe: Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>> =
            Symbolic::Concrete(ReplicatedTensor {
                shares: [
                    [
                        SymbolicHandle { op: "x00".into() }.into(),
                        SymbolicHandle { op: "x10".into() }.into(),
                    ],
                    [
                        SymbolicHandle { op: "x11".into() }.into(),
                        SymbolicHandle { op: "x21".into() }.into(),
                    ],
                    [
                        SymbolicHandle { op: "x22".into() }.into(),
                        SymbolicHandle { op: "x02".into() }.into(),
                    ],
                ],
            });

        let ye: Symbolic<ReplicatedTensor<Symbolic<Ring64Tensor>>> =
            Symbolic::Concrete(ReplicatedTensor {
                shares: [
                    [
                        SymbolicHandle { op: "y00".into() }.into(),
                        SymbolicHandle { op: "y10".into() }.into(),
                    ],
                    [
                        SymbolicHandle { op: "y11".into() }.into(),
                        SymbolicHandle { op: "y21".into() }.into(),
                    ],
                    [
                        SymbolicHandle { op: "y22".into() }.into(),
                        SymbolicHandle { op: "y02".into() }.into(),
                    ],
                ],
            });

        let ze = rep.add(&ctx, &xe, &ye);

        assert_eq!(
            ze,
            Symbolic::Concrete(ReplicatedTensor {
                shares: [
                    [
                        Symbolic::Symbolic(SymbolicHandle { op: "op_0".into() }),
                        Symbolic::Symbolic(SymbolicHandle { op: "op_1".into() }),
                    ],
                    [
                        Symbolic::Symbolic(SymbolicHandle { op: "op_2".into() }),
                        Symbolic::Symbolic(SymbolicHandle { op: "op_3".into() }),
                    ],
                    [
                        Symbolic::Symbolic(SymbolicHandle { op: "op_4".into() }),
                        Symbolic::Symbolic(SymbolicHandle { op: "op_5".into() }),
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
                        plc: HostPlacement {
                            player: "alice".into()
                        }
                        .into(),
                    }
                    .into(),
                    operands: vec!["x00".into(), "y00".into()],
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
                        plc: HostPlacement {
                            player: "alice".into()
                        }
                        .into(),
                    }
                    .into(),
                    operands: vec!["x10".into(), "y10".into()],
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
                        plc: HostPlacement {
                            player: "bob".into()
                        }
                        .into(),
                    }
                    .into(),
                    operands: vec!["x11".into(), "y11".into()],
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
                        plc: HostPlacement {
                            player: "bob".into()
                        }
                        .into(),
                    }
                    .into(),
                    operands: vec!["x21".into(), "y21".into()],
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
                        plc: HostPlacement {
                            player: "carole".into()
                        }
                        .into(),
                    }
                    .into(),
                    operands: vec!["x22".into(), "y22".into()],
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
                        plc: HostPlacement {
                            player: "carole".into()
                        }
                        .into(),
                    }
                    .into(),
                    operands: vec!["x02".into(), "y02".into()],
                },
            ]
        );
    }

    #[test]
    fn test_rep_share_concrete() {
        let rep = ReplicatedPlacement {
            players: ["alice".into(), "bob".into(), "carole".into()],
        };

        let ctx = ConcreteContext::default();

        let x: Ring64Tensor = RingTensor(5);
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
        println!("SYMBOLIC {:?}", ze);
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
                    plc: alice_plc.clone().into(),
                }
                .into(),
                operands: vec![],
            },
            Operation {
                name: "xe".into(),
                operator: RepShareOp {
                    sig: UnarySignature {
                        arg0: Ty::Ring128Tensor,
                        ret: Ty::Replicated128Tensor,
                    }
                    .into(),
                    plc: rep_plc.clone().into(),
                }
                .into(),
                operands: vec!["x".into()],
            },
            Operation {
                name: "y".into(),
                operator: RingSampleOp {
                    sig: NullarySignature {
                        ret: Ty::Ring128Tensor,
                    }
                    .into(),
                    plc: bob_plc.clone().into(),
                }
                .into(),
                operands: vec![],
            },
            Operation {
                name: "ye".into(),
                operator: RepShareOp {
                    sig: UnarySignature {
                        arg0: Ty::Ring128Tensor,
                        ret: Ty::Replicated128Tensor,
                    }
                    .into(),
                    plc: rep_plc.clone().into(),
                }
                .into(),
                operands: vec!["y".into()],
            },
            Operation {
                name: "s".into(),
                operator: RepSetupOp {
                    sig: NullarySignature {
                        ret: Ty::ReplicatedSetup,
                    }
                    .into(),
                    plc: rep_plc.clone().into(),
                }
                .into(),
                operands: vec![],
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
                    plc: rep_plc.clone().into(),
                }
                .into(),
                operands: vec!["s".into(), "xe".into(), "ye".into()],
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
                    plc: rep_plc.clone().into(),
                }
                .into(),
                operands: vec!["s".into(), "xe".into(), "ye".into()],
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
            let res = ctx.execute(operator, operands);
            env.insert(op.name.clone(), res);
        }

        println!("{:?}\n\n", env);

        let ctx = ConcreteContext::default();
        let mut env: HashMap<String, Value> = HashMap::default();

        for op in ops.iter() {
            let operator = op.operator.clone();
            let operands = op
                .operands
                .iter()
                .map(|input_name| env.get(input_name).unwrap().clone())
                .collect();
            let res = ctx.execute(operator, operands);
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
                    plc: alice_plc.clone().into(),
                }
                .into(),
                operands: vec![],
            },
            Operation {
                name: "xe".into(),
                operator: RepShareOp {
                    sig: UnarySignature {
                        arg0: Ty::BitTensor,
                        ret: Ty::ReplicatedBitTensor,
                    }
                    .into(),
                    plc: rep_plc.clone().into(),
                }
                .into(),
                operands: vec!["x".into()],
            },
            Operation {
                name: "y".into(),
                operator: BitSampleOp {
                    sig: NullarySignature { ret: Ty::BitTensor }.into(),
                    plc: bob_plc.clone().into(),
                }
                .into(),
                operands: vec![],
            },
            Operation {
                name: "ye".into(),
                operator: RepShareOp {
                    sig: UnarySignature {
                        arg0: Ty::BitTensor,
                        ret: Ty::ReplicatedBitTensor,
                    }
                    .into(),
                    plc: rep_plc.clone().into(),
                }
                .into(),
                operands: vec!["y".into()],
            },
            Operation {
                name: "s".into(),
                operator: RepSetupOp {
                    sig: NullarySignature {
                        ret: Ty::ReplicatedSetup,
                    }
                    .into(),
                    plc: rep_plc.clone().into(),
                }
                .into(),
                operands: vec![],
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
                    plc: rep_plc.clone().into(),
                }
                .into(),
                operands: vec!["s".into(), "xe".into(), "ye".into()],
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
                    plc: rep_plc.clone().into(),
                }
                .into(),
                operands: vec!["s".into(), "xe".into(), "ye".into()],
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
            let res = ctx.execute(operator, operands);
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
            let res = ctx.execute(operator, operands);
            env.insert(op.name.clone(), res);
        }

        println!("{:?}", env);
    }
}
