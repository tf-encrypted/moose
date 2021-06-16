#![allow(dead_code)]
#![allow(unused_variables)]

use macros::with_context;
use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::ops::{Add, Mul, Sub};
use std::ops::{BitAnd, BitXor};

#[derive(Debug, Clone, PartialEq)]
pub enum Placement {
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
pub struct HostPlacement {
    player: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReplicatedPlacement {
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
pub enum PlacementTy {
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
    Fixed64Tensor,
    Fixed128Tensor,
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
    const TY: Ty;
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
    ReplicatedSetup(ReplicatedSetup),
    PrfKey(PrfKey),
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
            Value::ReplicatedSetup(_) => Ty::ReplicatedSetup,
            Value::PrfKey(_) => Ty::PrfKey,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SymbolicValue {
    Fixed64Tensor(<Fixed64Tensor as KnownType>::Symbolic),
    Fixed128Tensor(<Fixed128Tensor as KnownType>::Symbolic),
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
    ReplicatedSetup,
    Symbolic<AbstractReplicatedSetup<<PrfKey as KnownType>::Symbolic>>
);
value!(PrfKey, Symbolic<PrfKey>);

#[derive(Clone, Debug, PartialEq)]
pub enum Symbolic<T: Placed> {
    Symbolic(SymbolicHandle<T::Placement>),
    Concrete(T),
}

pub trait Placed {
    type Placement;

    fn placement(&self) -> Self::Placement;
}

impl Placed for BitTensor {
    type Placement = HostPlacement;

    fn placement(&self) -> Self::Placement {
        self.1.clone()
    }
}

impl<T> Placed for RingTensor<T> {
    type Placement = HostPlacement;

    fn placement(&self) -> Self::Placement {
        self.1.clone()
    }
}

impl<R> Placed for ReplicatedTensor<R>
where
    R: Placed<Placement = HostPlacement>,
{
    type Placement = ReplicatedPlacement;

    fn placement(&self) -> Self::Placement {
        let ReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = self;

        let player0 = x00.placement();
        assert_eq!(x10.placement(), player0);

        let player1 = x11.placement();
        assert_eq!(x21.placement(), player1);

        let player2 = x22.placement();
        assert_eq!(x02.placement(), player2);

        let players = [player0.player, player1.player, player2.player];
        ReplicatedPlacement { players }
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

impl Placed for PrfKey {
    type Placement = HostPlacement;

    fn placement(&self) -> Self::Placement {
        self.1.clone()
    }
}

impl<K> Placed for AbstractReplicatedSetup<K>
where
    K: Placed<Placement = HostPlacement>,
{
    type Placement = ReplicatedPlacement;

    fn placement(&self) -> Self::Placement {
        let AbstractReplicatedSetup {
            keys: [[x00, x10], [x11, x21], [x22, x02]],
        } = self;

        let player0 = x00.placement();
        assert_eq!(x10.placement(), player0);

        let player1 = x11.placement();
        assert_eq!(x21.placement(), player1);

        let player2 = x22.placement();
        assert_eq!(x02.placement(), player2);

        let players = [player0.player, player1.player, player2.player];
        ReplicatedPlacement { players }
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
    FixedAddOp(FixedAddOp),
    FixedMulOp(FixedMulOp),
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
operator!(FixedAddOp);
operator!(FixedMulOp);

#[derive(Clone, Debug, PartialEq)]
struct Operation {
    name: String,
    operator: Operator,
    operands: Vec<String>,
    plc: Placement,
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
pub struct RingTensor<T>(T, HostPlacement);

impl Add<RingTensor<u64>> for RingTensor<u64> {
    type Output = RingTensor<u64>;

    fn add(self, other: RingTensor<u64>) -> Self::Output {
        RingTensor(self.0.wrapping_add(other.0), self.1)
    }
}

impl Add<RingTensor<u128>> for RingTensor<u128> {
    type Output = RingTensor<u128>;

    fn add(self, other: RingTensor<u128>) -> Self::Output {
        RingTensor(self.0.wrapping_add(other.0), self.1)
    }
}

impl Sub<RingTensor<u64>> for RingTensor<u64> {
    type Output = RingTensor<u64>;

    fn sub(self, other: RingTensor<u64>) -> Self::Output {
        RingTensor(self.0.wrapping_sub(other.0), self.1)
    }
}

impl Sub<RingTensor<u128>> for RingTensor<u128> {
    type Output = RingTensor<u128>;

    fn sub(self, other: RingTensor<u128>) -> Self::Output {
        RingTensor(self.0.wrapping_sub(other.0), self.1)
    }
}

impl Mul<RingTensor<u64>> for RingTensor<u64> {
    type Output = RingTensor<u64>;

    fn mul(self, other: RingTensor<u64>) -> Self::Output {
        RingTensor(self.0.wrapping_mul(other.0), self.1)
    }
}

impl Mul<RingTensor<u128>> for RingTensor<u128> {
    type Output = RingTensor<u128>;

    fn mul(self, other: RingTensor<u128>) -> Self::Output {
        RingTensor(self.0.wrapping_mul(other.0), self.1)
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

#[derive(Clone, Debug, PartialEq)]
pub struct ReplicatedTensor<R> {
    shares: [[R; 2]; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct PrfKey([u8; 16], HostPlacement);

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

pub type Fixed64Tensor = FixedTensor<Ring64Tensor, Replicated64Tensor>;

pub type Fixed128Tensor = FixedTensor<Ring128Tensor, Replicated128Tensor>;

#[derive(Clone, Debug, PartialEq)]
pub enum FixedTensor<RingTensorT, ReplicatedTensorT> {
    RingTensor(RingTensorT),
    ReplicatedTensor(ReplicatedTensorT),
}

macro_rules! modelled {
    /*
    Nullary
    */
    ($t:ident::$f:ident, $plc:ty, () -> $u:ty, $op:ident) => {
        impl NullaryKernelCheck<ConcreteContext, $plc, $u> for $op {
            fn check(ctx: &ConcreteContext, plc: &$plc) -> $u {
                // NOTE we shouldn't do anything here, the kernel call is simply to check

                // NOTE not sure whether to add `unimplemented!`. it might be better to
                // simply make sure the Check traits are private.
                <Self as NullaryKernel<ConcreteContext, $plc, $u>>::kernel(ctx, plc)
            }
        }

        impl $t<ConcreteContext, $u> for $plc {
            fn $f(&self, ctx: &ConcreteContext) -> $u {
                let sig = NullarySignature {
                    ret: <$u as KnownType>::TY,
                };
                let op = $op::from_signature(sig);
                ctx.execute(op.into(), &self.into(), vec![])
                    .try_into()
                    .unwrap()
            }
        }

        impl $t<SymbolicContext, <$u as KnownType>::Symbolic> for $plc {
            fn $f(&self, ctx: &SymbolicContext) -> <$u as KnownType>::Symbolic {
                let sig = NullarySignature {
                    ret: <$u as KnownType>::TY,
                };
                let op = $op::from_signature(sig);
                ctx.execute(op.into(), &self.into(), vec![])
                    .try_into()
                    .unwrap()
            }
        }
    };

    /*
    Unary
    */
    ($t:ident::$f:ident, $plc:ty, ($t0:ty) -> $u:ty, $op:ident) => {
        impl UnaryKernelCheck<ConcreteContext, $plc, $t0, $u> for $op {
            fn check(ctx: &ConcreteContext, plc: &$plc, x0: $t0) -> $u {
                <Self as UnaryKernel<ConcreteContext, $plc, $t0, $u>>::kernel(ctx, plc, x0)
            }
        }

        impl $t<ConcreteContext, $t0> for $plc {
            type Output = $u;

            fn $f(&self, ctx: &ConcreteContext, x0: &$t0) -> Self::Output {
                let sig = UnarySignature {
                    arg0: <$t0 as KnownType>::TY,
                    ret: <$u as KnownType>::TY,
                };
                let op = $op::from_signature(sig);
                ctx.execute(op.into(), &self.into(), vec![x0.clone().into()])
                    .try_into()
                    .unwrap()
            }
        }

        impl $t<SymbolicContext, <$t0 as KnownType>::Symbolic> for $plc {
            type Output = <$u as KnownType>::Symbolic;

            fn $f(&self, ctx: &SymbolicContext, x0: &<$t0 as KnownType>::Symbolic) -> Self::Output {
                let sig = UnarySignature {
                    arg0: <<$t0 as KnownType>::Symbolic as KnownType>::TY,
                    ret: <<$u as KnownType>::Symbolic as KnownType>::TY,
                };
                let op = $op::from_signature(sig);
                ctx.execute(op.into(), &self.into(), vec![x0.clone().into()])
                    .try_into()
                    .unwrap()
            }
        }
    };

    /*
    Binary
    */
    ($t:ident::$f:ident, $plc:ty, ($t0:ty, $t1:ty) -> $u:ty, $op:ident) => {
        impl BinaryKernelCheck<ConcreteContext, $plc, $t0, $t1, $u> for $op {
            fn check(ctx: &ConcreteContext, plc: &$plc, x0: $t0, x1: $t1) -> $u {
                <Self as BinaryKernel<ConcreteContext, $plc, $t0, $t1, $u>>::kernel(
                    ctx, plc, x0, x1,
                )
            }
        }

        impl $t<ConcreteContext, $t0, $t1> for $plc {
            type Output = $u;

            fn $f(&self, ctx: &ConcreteContext, x0: &$t0, x1: &$t1) -> Self::Output {
                let sig = BinarySignature {
                    arg0: <$t0 as KnownType>::TY,
                    arg1: <$t1 as KnownType>::TY,
                    ret: <$u as KnownType>::TY,
                };
                let op = $op::from_signature(sig);
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
                x0: &<$t0 as KnownType>::Symbolic,
                x1: &<$t1 as KnownType>::Symbolic,
            ) -> Self::Output {
                let sig = BinarySignature {
                    arg0: <<$t0 as KnownType>::Symbolic as KnownType>::TY,
                    arg1: <<$t1 as KnownType>::Symbolic as KnownType>::TY,
                    ret: <<$u as KnownType>::Symbolic as KnownType>::TY,
                };
                let op = $op::from_signature(sig);
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
    ($t:ident::$f:ident, $plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty, $op:ident) => {
        impl TernaryKernelCheck<ConcreteContext, $plc, $t0, $t1, $t2, $u> for $op {
            fn check(ctx: &ConcreteContext, plc: &$plc, x0: $t0, x1: $t1, x2: $t2) -> $u {
                <Self as TernaryKernel<ConcreteContext, $plc, $t0, $t1, $t2, $u>>::kernel(
                    ctx, plc, x0, x1, x2,
                )
            }
        }

        impl $t<ConcreteContext, $t0, $t1, $t2> for $plc {
            type Output = $u;

            fn $f(&self, ctx: &ConcreteContext, x0: &$t0, x1: &$t1, x2: &$t2) -> Self::Output {
                let sig = TernarySignature {
                    arg0: <$t0 as KnownType>::TY,
                    arg1: <$t1 as KnownType>::TY,
                    arg2: <$t2 as KnownType>::TY,
                    ret: <$u as KnownType>::TY,
                };
                let op = $op::from_signature(sig);
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
                let op = $op::from_signature(sig);
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

trait PlacementAdd<C: Context, T, U> {
    type Output;

    fn add(&self, ctx: &C, x: &T, y: &U) -> Self::Output;
}

trait PlacementSub<C: Context, T, U> {
    type Output;

    fn sub(&self, ctx: &C, x: &T, y: &U) -> Self::Output;
}

trait PlacementMul<C: Context, T, U> {
    type Output;

    fn mul(&self, ctx: &C, x: &T, y: &U) -> Self::Output;
}

trait PlacementXor<C: Context, T, U> {
    type Output;

    fn xor(&self, ctx: &C, x: &T, y: &U) -> Self::Output;
}

trait PlacementAnd<C: Context, T, U> {
    type Output;

    fn and(&self, ctx: &C, x: &T, y: &U) -> Self::Output;
}

trait PlacementMulSetup<C: Context, S, T, U> {
    type Output;

    fn mul(&self, ctx: &C, s: &S, x: &T, y: &U) -> Self::Output;
}

trait PlacementShare<C: Context, T> {
    type Output;

    fn share(&self, ctx: &C, x: &T) -> Self::Output;
}

trait PlacementReveal<C: Context, T> {
    type Output;

    fn reveal(&self, ctx: &C, x: &T) -> Self::Output;
}

trait PlacementSample<C: Context, O> {
    fn sample(&self, ctx: &C) -> O;
}

pub trait Context {
    type Value;
    fn execute(&self, op: Operator, plc: &Placement, operands: Vec<Self::Value>) -> Self::Value;

    type ReplicatedSetup;
    fn replicated_setup(&self, plc: &ReplicatedPlacement) -> &Self::ReplicatedSetup;
}

#[derive(Clone, Debug, Default)]
pub struct ConcreteContext {
    replicated_keys: HashMap<ReplicatedPlacement, ReplicatedSetup>,
}

impl Context for ConcreteContext {
    type Value = Value;

    fn execute(&self, op: Operator, plc: &Placement, operands: Vec<Value>) -> Value {
        match op {
            Operator::PrfKeyGenOp(op) => op.compile(self, plc)(operands),
            Operator::RingSampleOp(op) => op.compile(self, plc)(operands),
            Operator::BitSampleOp(op) => op.compile(self, plc)(operands),
            Operator::RingAddOp(op) => op.compile(self, plc)(operands),
            Operator::BitXorOp(op) => op.compile(self, plc)(operands),
            Operator::BitAndOp(op) => op.compile(self, plc)(operands),
            Operator::RingSubOp(op) => op.compile(self, plc)(operands),
            Operator::RingMulOp(op) => op.compile(self, plc)(operands),
            Operator::RepSetupOp(op) => op.compile(self, plc)(operands),
            Operator::RepShareOp(op) => op.compile(self, plc)(operands),
            Operator::RepRevealOp(op) => op.compile(self, plc)(operands),
            Operator::RepAddOp(op) => op.compile(self, plc)(operands),
            Operator::RepMulOp(op) => op.compile(self, plc)(operands),
            Operator::ConstantOp(op) => op.compile(self, plc)(operands),
            Operator::FixedAddOp(op) => op.compile(self, plc)(operands),
            Operator::FixedMulOp(op) => op.compile(self, plc)(operands),
        }
    }

    type ReplicatedSetup = ReplicatedSetup;
    fn replicated_setup(&self, plc: &ReplicatedPlacement) -> &Self::ReplicatedSetup {
        self.replicated_keys.get(plc).unwrap()
    }
}

use std::sync::{Arc, RwLock};

#[derive(Clone, Debug, Default)]
pub struct SymbolicContext {
    ops: Arc<RwLock<Vec<Operation>>>, // TODO use HashMap so we can do some consistency checks on the fly?
    replicated_keys:
        HashMap<ReplicatedPlacement, Symbolic<AbstractReplicatedSetup<Symbolic<PrfKey>>>>,
}

impl Context for SymbolicContext {
    type Value = SymbolicValue;

    fn execute(
        &self,
        op: Operator,
        plc: &Placement,
        operands: Vec<SymbolicValue>,
    ) -> SymbolicValue {
        match op {
            Operator::PrfKeyGenOp(op) => op.execute_symbolic(self, plc, operands),
            Operator::RingSampleOp(op) => op.execute_symbolic(self, plc, operands),
            Operator::BitSampleOp(op) => op.execute_symbolic(self, plc, operands),
            Operator::RingAddOp(op) => op.execute_symbolic(self, plc, operands),
            Operator::BitXorOp(op) => op.execute_symbolic(self, plc, operands),
            Operator::BitAndOp(op) => op.execute_symbolic(self, plc, operands),
            Operator::RingSubOp(op) => op.execute_symbolic(self, plc, operands),
            Operator::RingMulOp(op) => op.execute_symbolic(self, plc, operands),
            Operator::RepSetupOp(op) => op.execute_symbolic(self, plc, operands),
            Operator::RepShareOp(op) => op.execute_symbolic(self, plc, operands),
            Operator::RepRevealOp(op) => op.execute_symbolic(self, plc, operands),
            Operator::RepAddOp(op) => op.execute_symbolic(self, plc, operands),
            Operator::RepMulOp(op) => op.execute_symbolic(self, plc, operands),
            Operator::ConstantOp(op) => op.execute_symbolic(self, plc, operands),
            Operator::FixedAddOp(op) => op.execute_symbolic(self, plc, operands),
            Operator::FixedMulOp(op) => op.execute_symbolic(self, plc, operands),
        }
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

macro_rules! runtime_kernel {

    /*
    Nullaray
    */

    ($op:ty, [$( ($plc:ty, () -> $u:ty => $k:expr), )+]) => {
        $(
        impl NullaryKernel<ConcreteContext, $plc, $u> for $op {
            fn kernel(ctx: &ConcreteContext, plc: &$plc) -> $u {
                $k(ctx, plc)
            }
        }
        )+

        impl $op {
            pub fn compile(&self, ctx: &ConcreteContext, plc: &Placement) -> Box<dyn Fn(Vec<Value>) -> Value> {
                match (plc.ty(), self.sig) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Nullary(NullarySignature{
                                ret: <$u>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into().unwrap();
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

    ($op:ty, [$( ($plc:ty, ($t0:ty) -> $u:ty => $k:expr), )+]) => {
        $(
        impl UnaryKernel<ConcreteContext, $plc, $t0, $u> for $op {
            fn kernel(ctx: &ConcreteContext, plc: &$plc, x0: $t0) -> $u {
                $k(ctx, plc, x0)
            }
        }
        )+

        impl $op {
            pub fn compile(&self, ctx: &ConcreteContext, plc: &Placement) -> Box<dyn Fn(Vec<Value>) -> Value> {
                match (plc.ty(), self.sig) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Unary(UnarySignature{
                                arg0: <$t0>::TY,
                                ret: <$u>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into().unwrap();
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

    ($op:ty, [$( ($plc:ty, ($t0:ty, $t1:ty) -> $u:ty => $k:expr), )+]) => {
        $(
        impl BinaryKernel<ConcreteContext, $plc, $t0, $t1, $u> for $op {
            fn kernel(ctx: &ConcreteContext, plc: &$plc, x0: $t0, x1: $t1) -> $u {
                $k(ctx, plc, x0, x1)
            }
        }
        )+

        impl $op {
            pub fn compile(&self, ctx: &ConcreteContext, plc: &Placement) -> Box<dyn Fn(Vec<Value>) -> Value> {
                match (plc.ty(), self.sig) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Binary(BinarySignature{
                                arg0: <$t0>::TY,
                                arg1: <$t1>::TY,
                                ret: <$u>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into().unwrap();
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

    ($op:ty, [$( ($plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => $k:expr), )+]) => {
        $(
        impl TernaryKernel<ConcreteContext, $plc, $t0, $t1, $t2, $u> for $op {
            fn kernel(ctx: &ConcreteContext, plc: &$plc, x0: $t0, x1: $t1, x2: $t2) -> $u {
                $k(ctx, plc, x0, x1, x2)
            }
        }
        )+

        impl $op {
            pub fn compile(&self, ctx: &ConcreteContext, plc: &Placement) -> Box<dyn Fn(Vec<Value>) -> Value> {
                match (plc.ty(), self.sig) {
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
                            let plc: $plc = plc.clone().try_into().unwrap();
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

    ($op:ty, [$( ($plc:ty, () -> $u:ty => $k:expr), )+]) => {
        impl $op {
            pub fn execute_symbolic(&self, ctx: &SymbolicContext, plc: &Placement,  _operands: Vec<SymbolicValue>) -> SymbolicValue {
                match (plc.ty(), self.sig) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Nullary(NullarySignature{
                                ret: <<$u as KnownType>::Symbolic as KnownType>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into().unwrap();

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

    ($op:ty, [$( ($plc:ty, ($t0:ty) -> $u:ty => $k:expr), )+]) => {
        impl $op {
            pub fn execute_symbolic(&self, ctx: &SymbolicContext, plc: &Placement, operands: Vec<SymbolicValue>) -> SymbolicValue {
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

    ($op:ty, [$( ($plc:ty, ($t0:ty, $t1:ty) -> $u:ty => $k:expr), )+]) => {
        impl $op {
            pub fn execute_symbolic(
                &self,
                ctx: &SymbolicContext,
                plc: &Placement,
                operands: Vec<SymbolicValue>,
            ) -> SymbolicValue {
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

    ($op:ty, [$( ($plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => $k:expr), )+]) => {
        impl $op {
            pub fn execute_symbolic(
                &self,
                ctx: &SymbolicContext,
                plc: &Placement,
                operands: Vec<SymbolicValue>,
            ) -> SymbolicValue {
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

    ($op:ty, [$( ($plc:ty, () -> $u:ty => $k:expr), )+]) => {
        runtime_kernel!($op, [$( ($plc, () -> $u => $k), )+]);
        compiletime_kernel!($op, [$( ($plc, () -> $u => |op, ctx, plc| {
            let op_name = ctx.add_operation(op, &[], &plc.clone().into());
            Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.into() })
        }), )+]);
    };

    /*
    Unary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty) -> $u:ty => $k:expr), )+]) => {
        runtime_kernel!($op, [$( ($plc, ($t0) -> $u => $k), )+]);
        compiletime_kernel!($op, [$( ($plc, ($t0) -> $u => |op, ctx, plc, x0| {
            let x0_op = match x0 {
                Symbolic::Symbolic(h) => h.op,
                Symbolic::Concrete(_) => unimplemented!(),
            };

            let op_name = ctx.add_operation(op, &[&x0_op], &plc.clone().into());
            Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.into() })
        }), )+]);
    };

    /*
    Binary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty, $t1:ty) -> $u:ty => $k:expr), )+]) => {
        runtime_kernel!($op, [$( ($plc, ($t0, $t1) -> $u => $k), )+]);
        compiletime_kernel!($op, [$( ($plc, ($t0, $t1) -> $u => |op, ctx, plc, x0, x1| {
            let x0_op = match x0 {
                Symbolic::Symbolic(h) => h.op,
                Symbolic::Concrete(_) => unimplemented!(),
            };

            let x1_op = match x1 {
                Symbolic::Symbolic(h) => h.op,
                Symbolic::Concrete(_) => unimplemented!(),
            };

            let op_name = ctx.add_operation(op, &[&x0_op, &x1_op], &plc.clone().into());
            Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.into() })
        }), )+]);
    };

    /*
    Ternary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => $k:expr), )+]) => {
        runtime_kernel!($op, [$( ($plc, ($t0, $t1, $t2) -> $u => $k), )+]);
        compiletime_kernel!($op, [$( ($plc, ($t0, $t1, $t2) -> $u => |op, ctx, plc, x0, x1, x2| {
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

            let op_name = ctx.add_operation(op, &[&x0_op, &x1_op, &x2_op], &plc.clone().into());
            Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.into() })
        }), )+]);
    };
}

/// Kernel function maybe be evaluated in symbolic contexts
macro_rules! hybrid_kernel {

    /*
    Nullary
    */

    ($op:ty, [$( ($plc:ty, () -> $u:ty => $k:expr), )+]) => {
        runtime_kernel!($op, [$( ($plc, () -> $u => $k), )+]);
        compiletime_kernel!($op, [$( ($plc, () -> $u => |_op, ctx, plc| {
            let y = $k(ctx, &plc);
            y.into()
        }), )+]);
    };

    /*
    Unary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty) -> $u:ty => $k:expr), )+]) => {
        runtime_kernel!($op, [$( ($plc, ($t0) -> $u => $k), )+]);
        compiletime_kernel!($op, [$( ($plc, ($t0) -> $u => |op, ctx, plc, x0| {
            let v0 = x0.clone().try_into();

            match v0 {
                Ok(v0) => {
                    let y = $k(ctx, &plc, v0);
                    y.into()
                }
                _ => match x0 {
                    Symbolic::Symbolic(h0) => {
                        let op_name = ctx.add_operation(op, &[&h0.op], &plc.clone().into());
                        Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.into() })
                    }
                    _ => unimplemented!() // ok
                }
            }
        }), )+]);
    };

    /*
    Binary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty, $t1:ty) -> $u:ty => $k:expr), )+]) => {
        runtime_kernel!($op, [$( ($plc, ($t0, $t1) -> $u => $k), )+]);
        compiletime_kernel!($op, [$( ($plc, ($t0, $t1) -> $u => |op, ctx, plc, x0, x1| {
            let v0 = x0.clone().try_into();
            let v1 = x1.clone().try_into();

            match (v0, v1) {
                (Ok(v0), Ok(v1)) => {
                    let y = $k(ctx, &plc, v0, v1);
                    y.into()
                }
                _ => match (x0, x1) {
                    (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1)) => {
                        let op_name = ctx.add_operation(op, &[&h0.op, &h1.op], &plc.clone().into());
                        Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.into() })
                    }
                    _ => unimplemented!() // ok
                }
            }
        }), )+]);
    };

    /*
    Ternary
    */

    ($op:ty, [$( ($plc:ty, ($t0:ty, $t1:ty, $t2:ty) -> $u:ty => $k:expr), )+]) => {
        runtime_kernel!($op, [$( ($plc, ($t0, $t1, $t2) -> $u => $k), )+]);
        compiletime_kernel!($op, [$( ($plc, ($t0, $t1, $t2) -> $u => |op, ctx, plc, x0, x1, x2| {
            let v0 = x0.clone().try_into();
            let v1 = x1.clone().try_into();
            let v2 = x2.clone().try_into();

            match (v0, v1, v2) {
                (Ok(v0), Ok(v1), Ok(v2)) => {
                    let y = $k(ctx, &plc, v0, v1, v2);
                    y.into()
                }
                _ => match (x0, x1, x2) {
                    (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1), Symbolic::Symbolic(h2)) => {
                        let op_name = ctx.add_operation(op, &[&h0.op, &h1.op, &h2.op], &plc.clone().into());
                        Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.into() })
                    }
                    _ => unimplemented!() // ok
                }
            }
        }), )+]);
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

#[derive(Clone, Debug, PartialEq)]
pub struct RepSetupOp {
    sig: Signature,
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
        (ReplicatedPlacement, () -> ReplicatedSetup => Self::kernel),
    ]
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepAddOp {
    sig: Signature,
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

impl RepAddOp {
    fn from_signature(sig: BinarySignature) -> Self {
        RepAddOp { sig: sig.into() }
    }

    fn rep_rep_kernel<C: Context, R>(
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

        let z00 = with_context!(player0, ctx, x00 + y00);
        let z10 = with_context!(player0, ctx, x10 + y10);

        let z11 = with_context!(player1, ctx, x11 + y11);
        let z21 = with_context!(player1, ctx, x21 + y21);

        let z22 = with_context!(player2, ctx, x22 + y22);
        let z02 = with_context!(player2, ctx, x02 + y02);

        ReplicatedTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
    }

    fn ring_rep_kernel<C: Context, R: KnownType>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        x: R,
        y: ReplicatedTensor<R>,
    ) -> ReplicatedTensor<R>
    where
        R: Clone,
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<C, R, R, Output = R>,
    {
        let (player0, player1, player2) = rep.host_placements();
        let x_plc = x.placement();

        let ReplicatedTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = y;

        let shares = match x_plc {
            _ if x_plc == player0 => {
                // add x to y0
                [
                    [with_context!(player0, ctx, x + y00), y10],
                    [y11, y21],
                    [y22, with_context!(player2, ctx, x + y02)],
                ]
            }
            _ if x_plc == player1 => {
                // add x to y1
                [
                    [y00, with_context!(player0, ctx, x + y10)],
                    [with_context!(player1, ctx, x + y11), y21],
                    [y22, y02],
                ]
            }
            _ if x_plc == player2 => {
                // add x to y2
                [
                    [y00, y10],
                    [y11, with_context!(player1, ctx, x + y21)],
                    [with_context!(player2, ctx, x + y22), y02],
                ]
            }
            _ => {
                // add x to y0; we could randomize this
                [
                    [with_context!(player0, ctx, x + y00), y10],
                    [y11, y21],
                    [y22, with_context!(player2, ctx, x + y02)],
                ]
            }
        };

        ReplicatedTensor { shares }
    }

    fn rep_ring_kernel<C: Context, R>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        x: ReplicatedTensor<R>,
        y: R,
    ) -> ReplicatedTensor<R>
    where
        R: Clone,
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<C, R, R, Output = R>,
    {
        let (player0, player1, player2) = rep.host_placements();
        let y_plc = y.placement();

        let ReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = x;

        let shares = match y_plc {
            _ if y_plc == player0 => {
                // add y to x0
                [
                    [with_context!(player0, ctx, x00 + y), x10],
                    [x11, x21],
                    [x22, with_context!(player2, ctx, x02 + y)],
                ]
            }
            _ if y_plc == player1 => {
                // add y to x1
                [
                    [x00, with_context!(player0, ctx, x10 + y)],
                    [with_context!(player1, ctx, x11 + y), x21],
                    [x22, x02],
                ]
            }
            _ if y_plc == player2 => {
                // add y to x2
                [
                    [x00, x10],
                    [x11, with_context!(player1, ctx, x21 + y)],
                    [with_context!(player2, ctx, x22 + y), x02],
                ]
            }
            _ => {
                // add y to x0; we could randomize this
                [
                    [with_context!(player0, ctx, x00 + y), x10],
                    [x11, x21],
                    [x22, with_context!(player2, ctx, x02 + y)],
                ]
            }
        };

        ReplicatedTensor { shares }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepMulOp {
    sig: Signature,
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

impl RepMulOp {
    fn from_signature(sig: TernarySignature) -> Self {
        RepMulOp { sig: sig.into() }
    }

    fn rep_rep_kernel<C: Context, R, K>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        setup: AbstractReplicatedSetup<K>,
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
        } = rep.zero_share(ctx, &setup);

        let z0 = with_context!(player0, ctx, { x00 * y00 + x00 * y10 + x10 * y00 + a0 });
        let z1 = with_context!(player1, ctx, { x11 * y11 + x11 * y21 + x21 * y11 + a1 });
        let z2 = with_context!(player2, ctx, { x22 * y22 + x22 * y02 + x02 * y22 + a2 });

        ReplicatedTensor {
            shares: [[z0.clone(), z1.clone()], [z1, z2.clone()], [z2, z0]],
        }
    }

    fn ring_rep_kernel<C: Context, R, K>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        _setup: AbstractReplicatedSetup<K>,
        x: R,
        y: ReplicatedTensor<R>,
    ) -> ReplicatedTensor<R>
    where
        HostPlacement: PlacementMul<C, R, R, Output = R>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let ReplicatedTensor {
            shares: [[y00, y10], [y11, y21], [y22, y02]],
        } = &y;

        let z00 = with_context!(player0, ctx, x * y00);
        let z10 = with_context!(player0, ctx, x * y10);

        let z11 = with_context!(player1, ctx, x * y11);
        let z21 = with_context!(player1, ctx, x * y21);

        let z22 = with_context!(player2, ctx, x * y22);
        let z02 = with_context!(player2, ctx, x * y02);

        ReplicatedTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
    }

    fn rep_ring_kernel<C: Context, R, K>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        _setup: AbstractReplicatedSetup<K>,
        x: ReplicatedTensor<R>,
        y: R,
    ) -> ReplicatedTensor<R>
    where
        HostPlacement: PlacementMul<C, R, R, Output = R>,
    {
        let (player0, player1, player2) = rep.host_placements();

        let ReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &x;

        let z00 = with_context!(player0, ctx, x00 * y);
        let z10 = with_context!(player0, ctx, x10 * y);

        let z11 = with_context!(player1, ctx, x11 * y);
        let z21 = with_context!(player1, ctx, x21 * y);

        let z22 = with_context!(player2, ctx, x22 * y);
        let z02 = with_context!(player2, ctx, x02 * y);

        ReplicatedTensor {
            shares: [[z00, z10], [z11, z21], [z22, z02]],
        }
    }
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
        let alpha0 = with_context!(player0, ctx, r00 - r10);

        let r11 = player1.sample(ctx);
        let r21 = player1.sample(ctx);
        let alpha1 = with_context!(player1, ctx, r11 - r21);

        let r22 = player2.sample(ctx);
        let r02 = player2.sample(ctx);
        let alpha2 = with_context!(player2, ctx, r22 - r02);

        ReplicatedZeroShare {
            alphas: [alpha0, alpha1, alpha2],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepShareOp {
    sig: Signature,
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

impl RepShareOp {
    fn from_signature(sig: UnarySignature) -> Self {
        RepShareOp { sig: sig.into() }
    }

    fn kernel<C: Context, R: Clone>(ctx: &C, rep: &ReplicatedPlacement, x: R) -> ReplicatedTensor<R>
    where
        R: Into<C::Value> + TryFrom<C::Value> + 'static,
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementSample<C, R>,
        HostPlacement: PlacementAdd<C, R, R, Output = R>,
        HostPlacement: PlacementSub<C, R, R, Output = R>,
    {
        let owner = x.placement();

        let x0 = owner.sample(ctx);
        let x1 = owner.sample(ctx);
        let x2 = with_context!(owner, ctx, x - (x0 + x1));

        ReplicatedTensor {
            shares: [[x0.clone(), x1.clone()], [x1, x2.clone()], [x2, x0]],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepRevealOp {
    sig: Signature,
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

impl RepRevealOp {
    fn from_signature(sig: UnarySignature) -> Self {
        RepRevealOp { sig: sig.into() }
    }

    fn kernel<C: Context, R: Clone>(ctx: &C, plc: &HostPlacement, xe: ReplicatedTensor<R>) -> R
    where
        R: Clone + 'static,
        HostPlacement: PlacementAdd<C, R, R, Output = R>,
    {
        let ReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &xe;

        with_context!(plc, ctx, x00 + x10 + x21)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RingAddOp {
    sig: Signature,
}

// NOTE uncomment the next line to see the kernel check system in action
// modelled!(PlacementAdd, HostPlacement, (Ring32Tensor, Ring32Tensor) -> Ring32Tensor, RingAddOp);
// NOTE that supporting op attributes might be a simple adding an ctor input to the macro: (Placement, Signature) -> Op
modelled!(PlacementAdd::add, HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor, RingAddOp);
modelled!(PlacementAdd::add, HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor, RingAddOp);

kernel! {
    RingAddOp,
    [
        (HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor => Self::kernel),
        (HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor => Self::kernel),
    ]
}

impl RingAddOp {
    fn from_signature(sig: BinarySignature) -> Self {
        RingAddOp { sig: sig.into() }
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
pub struct RingSubOp {
    sig: Signature,
}

modelled!(PlacementSub::sub, HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor, RingSubOp);
modelled!(PlacementSub::sub, HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor, RingSubOp);

kernel! {
    RingSubOp,
    [
        (HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor => Self::kernel),
        (HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor => Self::kernel),
    ]
}

impl RingSubOp {
    fn from_signature(sig: BinarySignature) -> Self {
        RingSubOp { sig: sig.into() }
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

#[derive(Clone, Debug, PartialEq)]
pub struct RingMulOp {
    sig: Signature,
}

modelled!(PlacementMul::mul, HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor, RingMulOp);
modelled!(PlacementMul::mul, HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor, RingMulOp);

kernel! {
    RingMulOp,
    [
        (HostPlacement, (Ring64Tensor, Ring64Tensor) -> Ring64Tensor => Self::kernel),
        (HostPlacement, (Ring128Tensor, Ring128Tensor) -> Ring128Tensor => Self::kernel),
    ]
}

impl RingMulOp {
    fn from_signature(sig: BinarySignature) -> Self {
        RingMulOp { sig: sig.into() }
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
pub struct BitXorOp {
    sig: Signature,
}

modelled!(PlacementXor::xor, HostPlacement, (BitTensor, BitTensor) -> BitTensor, BitXorOp);
modelled_alias!(PlacementAdd::add, HostPlacement, (BitTensor, BitTensor) -> BitTensor => PlacementXor::xor); // add = xor in Z2
modelled_alias!(PlacementSub::sub, HostPlacement, (BitTensor, BitTensor) -> BitTensor => PlacementXor::xor); // sub = xor in Z2

kernel! {
    BitXorOp,
    [
        (HostPlacement, (BitTensor, BitTensor) -> BitTensor => Self::kernel),
    ]
}

impl BitXorOp {
    fn from_signature(sig: BinarySignature) -> Self {
        BitXorOp { sig: sig.into() }
    }

    fn kernel<C: Context>(_ctx: &C, _plc: &HostPlacement, x: BitTensor, y: BitTensor) -> BitTensor
    where
        BitTensor: BitXor<BitTensor, Output = BitTensor>,
    {
        x ^ y
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BitAndOp {
    sig: Signature,
}

impl BitAndOp {
    fn from_signature(sig: BinarySignature) -> Self {
        BitAndOp { sig: sig.into() }
    }

    fn kernel<C: Context>(_ctx: &C, _plc: &HostPlacement, x: BitTensor, y: BitTensor) -> BitTensor
    where
        BitTensor: BitAnd<BitTensor, Output = BitTensor>,
    {
        x & y
    }
}

modelled!(PlacementAnd::and, HostPlacement, (BitTensor, BitTensor) -> BitTensor, BitAndOp);
modelled_alias!(PlacementMul::mul, HostPlacement, (BitTensor, BitTensor) -> BitTensor => PlacementAnd::and); // mul = and in Z2

kernel! {
    BitAndOp,
    [
        (HostPlacement, (BitTensor, BitTensor) -> BitTensor => Self::kernel),
    ]
}

trait PlacementKeyGen<C: Context, K> {
    fn keygen(&self, ctx: &C) -> K;
}

#[derive(Clone, Debug, PartialEq)]
pub struct PrfKeyGenOp {
    sig: Signature,
}

modelled!(PlacementKeyGen::keygen, HostPlacement, () -> PrfKey, PrfKeyGenOp);

kernel! {
    PrfKeyGenOp,
    [
        (HostPlacement, () -> PrfKey => Self::kernel),
    ]
}

impl PrfKeyGenOp {
    fn from_signature(sig: NullarySignature) -> Self {
        PrfKeyGenOp { sig: sig.into() }
    }

    fn kernel(ctx: &ConcreteContext, plc: &HostPlacement) -> PrfKey {
        // TODO
        PrfKey(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            plc.clone(),
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RingSampleOp {
    sig: Signature,
}

modelled!(PlacementSample::sample, HostPlacement, () -> Ring64Tensor, RingSampleOp);
modelled!(PlacementSample::sample, HostPlacement, () -> Ring128Tensor, RingSampleOp);

kernel! {
    RingSampleOp,
    [
        (HostPlacement, () -> Ring64Tensor => Self::kernel),
        (HostPlacement, () -> Ring128Tensor => Self::kernel),
    ]
}

impl RingSampleOp {
    fn from_signature(sig: NullarySignature) -> Self {
        RingSampleOp { sig: sig.into() }
    }

    fn kernel<T>(ctx: &ConcreteContext, plc: &HostPlacement) -> RingTensor<T>
    where
        T: From<u32>,
    {
        // TODO
        RingTensor::<T>(T::from(987654321), plc.clone())
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
    fn from_signature(sig: NullarySignature) -> Self {
        BitSampleOp { sig: sig.into() }
    }

    fn kernel(ctx: &ConcreteContext, plc: &HostPlacement) -> BitTensor {
        // TODO
        BitTensor(0, plc.clone())
    }
}

// TODO clippy complains if ConstantOp holds a Value but not
// sure where to introduce eg a Box: here or in Value itself?
// leaning towards Value
#[derive(Clone, Debug, PartialEq)]
pub struct ConstantOp {
    sig: Signature,
    val: Box<Value>,
}

impl ConstantOp {
    pub fn compile(
        &self,
        _ctx: &ConcreteContext,
        plc: &Placement,
    ) -> Box<dyn Fn(Vec<Value>) -> Value> {
        let val = self.val.clone();

        match plc {
            Placement::HostPlacement(_) => Box::new(move |_operands| -> Value { *val.clone() }),
            _ => unimplemented!(), // ok
        }
    }

    pub fn execute_symbolic(
        &self,
        ctx: &SymbolicContext,
        plc: &Placement,
        _operands: Vec<SymbolicValue>,
    ) -> SymbolicValue {
        match plc {
            Placement::HostPlacement(_) => {
                let op_name = ctx.add_operation(self, &[], plc);
                self.val
                    .ty()
                    .synthesize_symbolic_value(op_name, plc.clone())
            }
            _ => unimplemented!(), // ok
        }
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
    fn from_signature(sig: BinarySignature) -> Self {
        FixedMulOp { sig: sig.into() }
    }

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
    fn from_signature(sig: BinarySignature) -> Self {
        FixedAddOp { sig: sig.into() }
    }

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
}
