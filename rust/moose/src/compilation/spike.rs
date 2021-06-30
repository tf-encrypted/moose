#![allow(dead_code)]
#![allow(unused_variables)]

use macros::with_context;
use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::ops::{Add, Mul, Neg, Shl, Shr, Sub};
use std::ops::{BitAnd, BitXor};

#[derive(Debug, Clone, PartialEq)]
pub enum Placement {
    HostPlacement(HostPlacement),
    ReplicatedPlacement(ReplicatedPlacement),
    AdditivePlacement(AdditivePlacement),
}

impl Placement {
    pub fn ty(&self) -> PlacementTy {
        match self {
            Placement::HostPlacement(plc) => plc.ty(),
            Placement::ReplicatedPlacement(plc) => plc.ty(),
            Placement::AdditivePlacement(plc) => plc.ty(),
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AdditivePlacement {
    players: [String; 2],
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

impl AdditivePlacement {
    pub fn host_placements(&self) -> (HostPlacement, HostPlacement) {
        let player0 = HostPlacement {
            player: self.players[0].clone(),
        };
        let player1 = HostPlacement {
            player: self.players[1].clone(),
        };
        (player0, player1)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PlacementTy {
    HostTy,
    ReplicatedTy,
    AdditiveTy,
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

impl KnownPlacement for AdditivePlacement {
    const TY: PlacementTy = PlacementTy::AdditiveTy;
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
placement!(AdditivePlacement);

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

impl<T> Placed for Ring<T> {
    type Placement = HostPlacement;
    fn placement(&self) -> Self::Placement {
        self.1.clone()
    }
}

impl Placed for Shape {
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

impl<R> Placed for AdditiveTensor<R>
where
    R: Placed<Placement = HostPlacement>,
{
    type Placement = AdditivePlacement;

    fn placement(&self) -> Self::Placement {
        let AdditiveTensor { shares: [x0, x1] } = self;

        let player0 = x0.placement();
        let player1 = x1.placement();

        let players = [player0.player, player1.player];
        AdditivePlacement { players }
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
#[allow(clippy::enum_variant_names)]
#[allow(clippy::large_enum_variant)]
pub enum Operator {
    PrfKeyGenOp(PrfKeyGenOp),
    RingAddOp(RingAddOp),
    RingShlOp(RingShlOp),
    RingShrOp(RingShrOp),
    BitXorOp(BitXorOp),
    BitAndOp(BitAndOp),
    RingSubOp(RingSubOp),
    RingNegOp(RingNegOp),
    RingMulOp(RingMulOp),
    RingSampleOp(RingSampleOp),
    FillOp(FillOp),
    OnesOp(OnesOp),
    BitSampleOp(BitSampleOp),
    RepSetupOp(RepSetupOp),
    RepAddOp(RepAddOp),
    RepMulOp(RepMulOp),
    RepToAddOp(RepToAddOp),
    RepShareOp(RepShareOp),
    RepRevealOp(RepRevealOp),
    RepTruncPrOp(RepTruncPrOp),
    AdditiveAddOp(AdditiveAddOp),
    AdditiveSubOp(AdditiveSubOp),
    AdditiveMulOp(AdditiveMulOp),
    AdditiveRevealOp(AdditiveRevealOp),
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
operator!(RingShlOp);
operator!(RingShrOp);
operator!(BitXorOp);
operator!(BitAndOp);
operator!(RingSubOp);
operator!(RingNegOp);
operator!(RingMulOp);
operator!(RingSampleOp);
operator!(FillOp);
operator!(OnesOp);
operator!(BitSampleOp);
operator!(RepSetupOp);
operator!(RepAddOp);
operator!(RepMulOp);
operator!(RepToAddOp);
operator!(RepShareOp);
operator!(RepRevealOp);
operator!(RepTruncPrOp);
operator!(AdditiveAddOp);
operator!(AdditiveSubOp);
operator!(AdditiveMulOp);
operator!(AdditiveRevealOp);
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

#[derive(Clone, Debug, PartialEq)]
pub struct Ring<T>(T, HostPlacement);

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

impl Shl<usize> for RingTensor<u64> {
    type Output = RingTensor<u64>;
    fn shl(self, other: usize) -> Self::Output {
        RingTensor(self.0.wrapping_shl(other as u32), self.1)
    }
}

impl Shl<usize> for RingTensor<u128> {
    type Output = RingTensor<u128>;
    fn shl(self, other: usize) -> Self::Output {
        RingTensor(self.0.wrapping_shl(other as u32), self.1)
    }
}

impl Shr<usize> for RingTensor<u64> {
    type Output = RingTensor<u64>;
    fn shr(self, other: usize) -> Self::Output {
        RingTensor(self.0.wrapping_shr(other as u32), self.1)
    }
}

impl Shr<usize> for RingTensor<u128> {
    type Output = RingTensor<u128>;
    fn shr(self, other: usize) -> Self::Output {
        RingTensor(self.0.wrapping_shr(other as u32), self.1)
    }
}

impl Neg for RingTensor<u64> {
    type Output = RingTensor<u64>;
    fn neg(self) -> Self::Output {
        RingTensor(self.0.wrapping_neg(), self.1)
    }
}

impl Neg for RingTensor<u128> {
    type Output = RingTensor<u128>;
    fn neg(self) -> Self::Output {
        RingTensor(self.0.wrapping_neg(), self.1)
    }
}

// impl RingTensor<u128> {
//     fn fill(el: u128, plc: HostPlacement) -> RingTensor<u128> {
//         RingTensor(el, plc)
//     }
// }

impl<T> RingTensor<T> {
    fn fill(el: T, plc: HostPlacement) -> RingTensor<T> {
        RingTensor(el, plc)
    }
}

// impl RingTensor<u64> {
//     fn fill(el: u64, plc: HostPlacement) -> RingTensor<u64> {
//         RingTensor(el, plc)
//     }
// }

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

#[derive(Clone, Debug, PartialEq)]
pub struct ReplicatedTensor<R> {
    shares: [[R; 2]; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdditiveTensor<R> {
    shares: [R; 2],
}

#[derive(Clone, Debug, PartialEq)]
pub struct PrfKey([u8; 16], HostPlacement);

#[derive(Clone, Debug, PartialEq)]
pub struct Shape(Vec<u8>, HostPlacement);

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

pub type Ring64 = Ring<u64>;

pub type Ring128 = Ring<u128>;

pub type Bit = Ring<u8>;

pub type Replicated64Tensor = ReplicatedTensor<Ring64Tensor>;

pub type Replicated128Tensor = ReplicatedTensor<Ring128Tensor>;

pub type Additive64Tensor = AdditiveTensor<Ring64Tensor>;

pub type Additive128Tensor = AdditiveTensor<Ring128Tensor>;

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

        impl $t<ConcreteContext, $t0, $u> for $plc {
            fn $f(&self, ctx: &ConcreteContext, $($($attr_id:$attr_ty),*,)? x0: &$t0) -> $u {
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

        impl $t<SymbolicContext, <$t0 as KnownType>::Symbolic, <$u as KnownType>::Symbolic> for $plc {

            fn $f(&self, ctx: &SymbolicContext, $($($attr_id:$attr_ty),*,)? x0: &<$t0 as KnownType>::Symbolic) -> <$u as KnownType>::Symbolic
            {
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

        impl $t<ConcreteContext, $t0, $t1, $u> for $plc {
            fn $f(&self, ctx: &ConcreteContext, $($($attr_id:$attr_ty),*,)? x0: &$t0, x1: &$t1) -> $u {
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

        impl $t<SymbolicContext, <$t0 as KnownType>::Symbolic, <$t1 as KnownType>::Symbolic, <$u as KnownType>::Symbolic>
            for $plc
        {

            fn $f(
                &self,
                ctx: &SymbolicContext,
                $($($attr_id:$attr_ty),*,)?
                x0: &<$t0 as KnownType>::Symbolic,
                x1: &<$t1 as KnownType>::Symbolic,
            ) ->  <$u as KnownType>::Symbolic
            {
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

        impl $t<ConcreteContext, $t0, $t1, $t2, $u> for $plc {

            fn $f(&self, ctx: &ConcreteContext, $($($attr_id:$attr_ty),*,)? x0: &$t0, x1: &$t1, x2: &$t2) -> $u {
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
                <$u as KnownType>::Symbolic,
            > for $plc
        {

            fn $f(
                &self,
                ctx: &SymbolicContext,
                $($($attr_id:$attr_ty),*,)?
                x0: &<$t0 as KnownType>::Symbolic,
                x1: &<$t1 as KnownType>::Symbolic,
                x2: &<$t2 as KnownType>::Symbolic,
            ) ->
            <$u as KnownType>::Symbolic
            {
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
        impl $src_t<ConcreteContext, $t0, $t1, $u> for $plc {
            fn $src_f(&self, ctx: &ConcreteContext, x0: &$t0, x1: &$t1) -> $u {
                $dst_t::$dst_f(self, ctx, x0, x1)
            }
        }

        impl
            $src_t<
                SymbolicContext,
                <$t0 as KnownType>::Symbolic,
                <$t1 as KnownType>::Symbolic,
                <$u as KnownType>::Symbolic,
            > for $plc
        {
            fn $src_f(
                &self,
                ctx: &SymbolicContext,
                x0: &<$t0 as KnownType>::Symbolic,
                x1: &<$t1 as KnownType>::Symbolic,
            ) -> <$u as KnownType>::Symbolic {
                $dst_t::$dst_f(self, ctx, x0, x1)
            }
        }
    };
}

trait PlacementAdd<C: Context, T, U, O> {
    fn add(&self, ctx: &C, x: &T, y: &U) -> O;
}

trait PlacementSub<C: Context, T, U, O> {
    fn sub(&self, ctx: &C, x: &T, y: &U) -> O;
}

trait PlacementNeg<C: Context, T, O> {
    fn neg(&self, ctx: &C, x: &T) -> O;
}

trait PlacementMul<C: Context, T, U, O> {
    fn mul(&self, ctx: &C, x: &T, y: &U) -> O;
}
trait PlacementShl<C: Context, T, O> {
    fn shl(&self, ctx: &C, amount: usize, x: &T) -> O;
}

trait PlacementShr<C: Context, T, O> {
    fn shr(&self, ctx: &C, amount: usize, x: &T) -> O;
}

trait PlacementXor<C: Context, T, U, O> {
    fn xor(&self, ctx: &C, x: &T, y: &U) -> O;
}

trait PlacementAnd<C: Context, T, U, O> {
    fn and(&self, ctx: &C, x: &T, y: &U) -> O;
}

trait PlacementFill<C: Context, S, O> {
    fn fill(&self, ctx: &C, value: Value, shape: &S) -> O;
}

trait PlacementOnes<C: Context, S, O> {
    fn ones(&self, ctx: &C, shape: &S) -> O;
}

trait PlacementMulSetup<C: Context, S, T, U, O> {
    fn mul(&self, ctx: &C, s: &S, x: &T, y: &U) -> O;
}

trait PlacementShare<C: Context, T, O> {
    fn share(&self, ctx: &C, x: &T) -> O;
}

trait PlacementReveal<C: Context, T, O> {
    fn reveal(&self, ctx: &C, x: &T) -> O;
}

trait PlacementSample<C: Context, O> {
    fn sample(&self, ctx: &C) -> O;
}

trait PlacementRepToAdd<C: Context, T, O> {
    fn rep_to_add(&self, ctx: &C, x: &T) -> O;
}

trait PlacementAddToRep<C: Context, T, O> {
    fn add_to_rep(&self, ctx: &C, x: &T) -> O;
}

trait PlacementTruncPr<C: Context, S, T, O> {
    fn trunc_pr(&self, ctx: &C, amount: usize, s: &S, x: &T) -> O;
}

pub trait Context {
    type Value;
    fn execute(&self, op: Operator, plc: &Placement, operands: Vec<Self::Value>) -> Self::Value;

    type ReplicatedSetup;
    fn replicated_setup(&self, plc: &ReplicatedPlacement) -> &Self::ReplicatedSetup;
}

pub struct ConcreteContext {
    replicated_keys: HashMap<ReplicatedPlacement, ReplicatedSetup>,
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
            Operator::RingNegOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::RingMulOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::RingShlOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::RingShrOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::RepSetupOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::RepShareOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::RepRevealOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::RepAddOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::RepMulOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::RepToAddOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::RepTruncPrOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::AdditiveAddOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::AdditiveSubOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::AdditiveMulOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::AdditiveRevealOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::ConstantOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::FixedAddOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::FixedMulOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::FillOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
            Operator::OnesOp(op) => DispatchKernel::compile(&op, self, plc)(operands),
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

macro_rules! derive_runtime_kernel {
    (nullary, custom |$op:ident| $kf:expr, $self:ident) => {
        {
            let kf: &dyn Fn(&Self) -> Box<dyn Fn(&_, &_,) -> _> = &|$op| $kf;
            kf($self)
        }
    };
    (unary, custom |$op:ident| $kf:expr, $self:ident) => {
        {
            let kf: &dyn Fn(&Self) -> Box<dyn Fn(&_, &_, _) -> _> = &|$op| $kf;
            kf($self)
        }
    };
    (binary, custom |$op:ident| $kf:expr, $self:ident) => {
        {
            let kf: &dyn Fn(&Self) -> Box<dyn Fn(&_, &_, _, _) -> _> = &|$op| $kf;
            kf($self)
        }
    };
    (ternary, custom |$op:ident| $kf:expr, $self:ident) => {
        {
            let kf: &dyn Fn(&Self) -> Box<dyn Fn(&_, &_, _, _, _) -> _> = &|$op| $kf;
            kf($self)
        }
    };

    (nullary, attributes[$($attr:ident)+] $k:expr, $self:ident) => {
        {
            $(
            let $attr = $self.$attr.clone();
            )+
            Box::new(move |ctx, plc| {
                $k(ctx, plc, $($attr.clone()),+)
            })
        }
    };
    (unary, attributes[$($attr:ident)+] $k:expr, $self:ident) => {
        {
            $(
            let $attr = $self.$attr.clone();
            )+
            Box::new(move |ctx, plc, x0| {
                $k(ctx, plc, $($attr.clone()),+, x0)
            })
        }
    };
    (binary, attributes[$($attr:ident)+] $k:expr, $self:ident) => {
        {
            $(
            let $attr = $self.$attr.clone();
            )+
            Box::new(move |ctx, plc, x0, x1| {
                $k(ctx, plc, $($attr.clone()),+, x0, x1)
            })
        }
    };
    (ternary, attributes[$($attr:ident)+] $k:expr, $self:ident) => {
        {
            $(
            let $attr = $self.$attr.clone();
            )+
            Box::new(move |ctx, plc, x0, x1, x2| {
                $k(ctx, plc, $($attr.clone()),+), x0, x1, x2
            })
        }
    };

    (nullary, $k:expr, $self:ident) => {
        Box::new($k)
    };
    (unary, $k:expr, $self:ident) => {
        Box::new($k)
    };
    (binary, $k:expr, $self:ident) => {
        Box::new($k)
    };
    (ternary, $k:expr, $self:ident) => {
        Box::new($k)
    };
}

macro_rules! concrete_dispatch_kernel {

    /*
    Nullaray
    */

    ($op:ty, [$( ($plc:ty, () -> $u:ty), )+]) => {
        impl DispatchKernel<ConcreteContext> for $op {
            fn compile<'c>(&self, ctx: &'c ConcreteContext, plc: &Placement) -> Box<dyn Fn(Vec<Value>) -> Value + 'c> {
                match (plc.ty(), self.sig) {
                    $(
                        (
                            <$plc>::TY,
                            Signature::Nullary(NullarySignature{
                                ret: <$u>::TY,
                            })
                        ) => {
                            let plc: $plc = plc.clone().try_into().unwrap();

                            let k = <$op as NullaryKernel<ConcreteContext, $plc, $u>>::compile(self, &ctx, &plc);

                            Box::new(move |operands: Vec<Value>| {
                                assert_eq!(operands.len(), 0);

                                let y: $u = k(&ctx, &plc);
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
        impl DispatchKernel<ConcreteContext> for $op {
            fn compile<'c>(&self, ctx: &'c ConcreteContext, plc: &Placement) -> Box<dyn Fn(Vec<Value>) -> Value + 'c> {
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

                            let k = <$op as UnaryKernel<ConcreteContext, $plc, $t0, $u>>::compile(self, &ctx, &plc);

                            Box::new(move |operands: Vec<Value>| {
                                assert_eq!(operands.len(), 1);

                                let x0: $t0 = operands.get(0).unwrap().clone().try_into().unwrap();

                                let y: $u = k(&ctx, &plc, x0);
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
        impl DispatchKernel<ConcreteContext> for $op {
            fn compile<'c>(&self, ctx: &'c ConcreteContext, plc: &Placement) -> Box<dyn Fn(Vec<Value>) -> Value + 'c> {
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

                            let k = <$op as BinaryKernel<
                                ConcreteContext,
                                $plc,
                                $t0,
                                $t1,
                                $u
                            >>::compile(self, &ctx, &plc);

                            Box::new(move |operands| -> Value {
                                assert_eq!(operands.len(), 2);

                                let x0: $t0 = operands.get(0).unwrap().clone().try_into().unwrap();
                                let x1: $t1 = operands.get(1).unwrap().clone().try_into().unwrap();

                                let y: $u = k(&ctx, &plc, x0, x1);
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
        impl DispatchKernel<ConcreteContext> for $op {
            fn compile<'c>(&self, ctx: &'c ConcreteContext, plc: &Placement) -> Box<dyn Fn(Vec<Value>) -> Value + 'c> {
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

                            let k = <$op as TernaryKernel<ConcreteContext, $plc, $t0, $t1, $t2, $u>>::compile(self, &ctx, &plc);

                            Box::new(move |operands: Vec<Value>| -> Value {
                                assert_eq!(operands.len(), 3);

                                let x0: $t0 = operands.get(0).unwrap().clone().try_into().unwrap();
                                let x1: $t1 = operands.get(1).unwrap().clone().try_into().unwrap();
                                let x2: $t2 = operands.get(2).unwrap().clone().try_into().unwrap();

                                let y: $u = k(&ctx, &plc, x0, x1, x2);
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

/// Kernel function is never used in symbolic contexts
macro_rules! kernel {

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
                fn compile(&self, ctx: &ConcreteContext, plc: &$plc) -> Box<dyn Fn(&ConcreteContext, &$plc) -> $u> {
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
                    let op = self.clone();
                    Box::new(move |
                        ctx: &SymbolicContext,
                        plc: &$plc,
                    | {
                        let op_name = ctx.add_operation(&op, &[], &plc.clone().into());
                        Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.clone().into() })
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
                    let op = self.clone();
                    Box::new(move |
                        ctx: &SymbolicContext,
                        plc: &$plc,
                        x0: <$t0 as KnownType>::Symbolic,
                    | {
                        match x0 {
                            Symbolic::Symbolic(h0) => {
                                let op_name = ctx.add_operation(&op, &[&h0.op], &plc.clone().into());
                                Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.clone().into() })
                            }
                            _ => unimplemented!()
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
                    let op = self.clone();
                    Box::new(move |
                        ctx: &SymbolicContext,
                        plc: &$plc,
                        x0: <$t0 as KnownType>::Symbolic,
                        x1: <$t1 as KnownType>::Symbolic,
                    | {
                        match (x0, x1) {
                            (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1)) => {
                                let op_name = ctx.add_operation(&op, &[&h0.op, &h1.op], &plc.clone().into());
                                Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.clone().into() })
                            }
                            _ => unimplemented!()
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
                    let op = self.clone();
                    Box::new(move |
                        ctx: &SymbolicContext,
                        plc: &$plc,
                        x0: <$t0 as KnownType>::Symbolic,
                        x1: <$t1 as KnownType>::Symbolic,
                        x2: <$t2 as KnownType>::Symbolic,
                    | {
                        match (x0, x1, x2) {
                            (Symbolic::Symbolic(h0), Symbolic::Symbolic(h1), Symbolic::Symbolic(h2)) => {
                                let op_name = ctx.add_operation(&op, &[&h0.op, &h1.op, &h2.op], &plc.clone().into());
                                Symbolic::Symbolic(SymbolicHandle { op: op_name, plc: plc.clone().into() })
                            }
                            _ => unimplemented!()
                        }
                    })
                }
            }
        )+
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

// TODO if rustc can't figure out how to optimize Box<dyn Fn...> for
// function kernels then we could consider returning an enum over
// fn.. and Box<dyn Fn...> in the traits below instead

pub trait NullaryKernel<C: Context, P, Y> {
    fn compile(&self, ctx: &C, plc: &P) -> Box<dyn Fn(&C, &P) -> Y>;
}

pub trait UnaryKernel<C: Context, P, X0, Y> {
    fn compile(&self, ctx: &C, plc: &P) -> Box<dyn Fn(&C, &P, X0) -> Y>;
}

pub trait BinaryKernel<C: Context, P, X0, X1, Y> {
    fn compile(&self, ctx: &C, plc: &P) -> Box<dyn Fn(&C, &P, X0, X1) -> Y>;
}

pub trait TernaryKernel<C: Context, P, X0, X1, X2, Y> {
    fn compile(&self, ctx: &C, plc: &P) -> Box<dyn Fn(&C, &P, X0, X1, X2) -> Y>;
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

pub trait DispatchKernel<C: Context> {
    fn compile<'c>(
        &self,
        ctx: &'c C,
        plc: &Placement,
    ) -> Box<dyn Fn(Vec<C::Value>) -> C::Value + 'c>;
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
pub struct RepToAddOp {
    sig: Signature,
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

impl RepToAddOp {
    fn rep_to_add_kernel<C: Context, R>(
        ctx: &C,
        add: &AdditivePlacement,
        x: ReplicatedTensor<R>,
    ) -> AdditiveTensor<R>
    where
        R: Clone,
        HostPlacement: PlacementAdd<C, R, R, R>,
        R: Placed<Placement = HostPlacement>,
    {
        let (player_a, player_b) = add.host_placements();
        let (player0, player1, player2) = x.placement().host_placements();

        let ReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = x;

        let shares = match () {
            _ if player_a == player0 && player_b == player1 => {
                [with_context!(player0, ctx, x00 + x10), x21]
            }
            _ if player_a == player0 && player_b == player2 => {
                [with_context!(player0, ctx, x00 + x10), x22]
            }
            _ if player_a == player1 && player_b == player2 => {
                [with_context!(player1, ctx, x11 + x21), x02]
            }
            _ if player_a == player1 && player_b == player0 => {
                [x21, with_context!(player0, ctx, x00 + x10)]
            }
            _ if player_a == player2 && player_b == player0 => {
                [x22, with_context!(player0, ctx, x00 + x10)]
            }
            _ => [with_context!(player_a, ctx, x00 + x10), x21],
        };
        AdditiveTensor { shares }
    }
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
    fn rep_rep_kernel<C: Context, R>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        x: ReplicatedTensor<R>,
        y: ReplicatedTensor<R>,
    ) -> ReplicatedTensor<R>
    where
        R: Clone,
        HostPlacement: PlacementAdd<C, R, R, R>,
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
        HostPlacement: PlacementAdd<C, R, R, R>,
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
        HostPlacement: PlacementAdd<C, R, R, R>,
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
pub struct AdditiveAddOp {
    sig: Signature,
}

modelled!(PlacementAdd::add, AdditivePlacement, (Additive64Tensor, Additive64Tensor) -> Additive64Tensor, AdditiveAddOp);
modelled!(PlacementAdd::add, AdditivePlacement, (Additive128Tensor, Additive128Tensor) -> Additive128Tensor, AdditiveAddOp);
modelled!(PlacementAdd::add, AdditivePlacement, (Additive64Tensor, Ring64Tensor) -> Additive64Tensor, AdditiveAddOp);
modelled!(PlacementAdd::add, AdditivePlacement, (Ring64Tensor, Additive64Tensor) -> Additive64Tensor, AdditiveAddOp);
modelled!(PlacementAdd::add, AdditivePlacement, (Additive128Tensor, Ring128Tensor) -> Additive128Tensor, AdditiveAddOp);
modelled!(PlacementAdd::add, AdditivePlacement, (Ring128Tensor, Additive128Tensor) -> Additive128Tensor, AdditiveAddOp);

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

impl AdditiveAddOp {
    fn add_add_kernel<C: Context, R>(
        ctx: &C,
        add: &AdditivePlacement,
        x: AdditiveTensor<R>,
        y: AdditiveTensor<R>,
    ) -> AdditiveTensor<R>
    where
        HostPlacement: PlacementAdd<C, R, R, R>,
    {
        let (player0, player1) = add.host_placements();

        let AdditiveTensor { shares: [x0, x1] } = &x;

        let AdditiveTensor { shares: [y0, y1] } = &y;

        let z0 = with_context!(player0, ctx, x0 + y0);
        let z1 = with_context!(player1, ctx, x1 + y1);

        AdditiveTensor { shares: [z0, z1] }
    }

    fn add_ring_kernel<C: Context, R>(
        ctx: &C,
        add: &AdditivePlacement,
        x: AdditiveTensor<R>,
        y: R,
    ) -> AdditiveTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<C, R, R, R>,
    {
        let (player0, player1) = add.host_placements();
        let AdditiveTensor { shares: [x0, x1] } = x;

        let y_plc = y.placement();

        let shares = match y_plc {
            _ if y_plc == player0 => [with_context!(player0, ctx, x0 + y), x1],
            _ if y_plc == player1 => [x0, with_context!(player1, ctx, x1 + y)],
            _ => [with_context!(player0, ctx, x0 + y), x1],
        };
        AdditiveTensor { shares }
    }

    fn ring_add_kernel<C: Context, R>(
        ctx: &C,
        add: &AdditivePlacement,
        x: R,
        y: AdditiveTensor<R>,
    ) -> AdditiveTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementAdd<C, R, R, R>,
    {
        let (player0, player1) = add.host_placements();
        let AdditiveTensor { shares: [y0, y1] } = y;

        let x_plc = x.placement();

        let shares = match x_plc {
            _ if x_plc == player0 => [with_context!(player0, ctx, y0 + x), y1],
            _ if x_plc == player1 => [y0, with_context!(player1, ctx, x + y1)],
            _ => [with_context!(player0, ctx, x + y0), y1],
        };
        AdditiveTensor { shares }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdditiveSubOp {
    sig: Signature,
}

modelled!(PlacementSub::sub, AdditivePlacement, (Additive64Tensor, Additive64Tensor) -> Additive64Tensor, AdditiveSubOp);
modelled!(PlacementSub::sub, AdditivePlacement, (Additive128Tensor, Additive128Tensor) -> Additive128Tensor, AdditiveSubOp);
modelled!(PlacementSub::sub, AdditivePlacement, (Additive64Tensor, Ring64Tensor) -> Additive64Tensor, AdditiveSubOp);
modelled!(PlacementSub::sub, AdditivePlacement, (Ring64Tensor, Additive64Tensor) -> Additive64Tensor, AdditiveSubOp);
modelled!(PlacementSub::sub, AdditivePlacement, (Additive128Tensor, Ring128Tensor) -> Additive128Tensor, AdditiveSubOp);
modelled!(PlacementSub::sub, AdditivePlacement, (Ring128Tensor, Additive128Tensor) -> Additive128Tensor, AdditiveSubOp);

hybrid_kernel! {
    AdditiveSubOp,
    [
        (AdditivePlacement, (Additive64Tensor, Additive64Tensor) -> Additive64Tensor => Self::add_add_kernel),
        (AdditivePlacement, (Additive128Tensor, Additive128Tensor) -> Additive128Tensor => Self::add_add_kernel),
        (AdditivePlacement, (Additive64Tensor, Ring64Tensor) -> Additive64Tensor => Self::add_ring_kernel),
        (AdditivePlacement, (Additive128Tensor, Ring128Tensor) -> Additive128Tensor => Self::add_ring_kernel),
        (AdditivePlacement, (Ring64Tensor, Additive64Tensor) -> Additive64Tensor => Self::ring_add_kernel),
        (AdditivePlacement, (Ring128Tensor, Additive128Tensor) -> Additive128Tensor => Self::ring_add_kernel),
    ]
}

impl AdditiveSubOp {
    fn add_add_kernel<C: Context, R>(
        ctx: &C,
        add: &AdditivePlacement,
        x: AdditiveTensor<R>,
        y: AdditiveTensor<R>,
    ) -> AdditiveTensor<R>
    where
        HostPlacement: PlacementSub<C, R, R, R>,
    {
        let (player0, player1) = add.host_placements();

        let AdditiveTensor { shares: [x0, x1] } = &x;

        let AdditiveTensor { shares: [y0, y1] } = &y;

        let z0 = with_context!(player0, ctx, x0 - y0);
        let z1 = with_context!(player1, ctx, x1 - y1);

        AdditiveTensor { shares: [z0, z1] }
    }

    fn add_ring_kernel<C: Context, R>(
        ctx: &C,
        add: &AdditivePlacement,
        x: AdditiveTensor<R>,
        y: R,
    ) -> AdditiveTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementSub<C, R, R, R>,
    {
        let (player0, player1) = add.host_placements();
        let AdditiveTensor { shares: [x0, x1] } = x;

        let y_plc = y.placement();

        let shares = match y_plc {
            _ if y_plc == player0 => [with_context!(player0, ctx, x0 - y), x1],
            _ if y_plc == player1 => [x0, with_context!(player1, ctx, x1 - y)],
            _ => [with_context!(player0, ctx, x0 - y), x1],
        };
        AdditiveTensor { shares }
    }

    fn ring_add_kernel<C: Context, R>(
        ctx: &C,
        add: &AdditivePlacement,
        x: R,
        y: AdditiveTensor<R>,
    ) -> AdditiveTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementSub<C, R, R, R>,
        HostPlacement: PlacementNeg<C, R, R>,
    {
        let (player0, player1) = add.host_placements();
        let AdditiveTensor { shares: [y0, y1] } = y;

        let x_plc = x.placement();
        let shares = match x_plc {
            _ if x_plc == player0 => [with_context!(player0, ctx, x - y0), player0.neg(ctx, &y1)],
            _ if x_plc == player1 => [player0.neg(ctx, &y0), with_context!(player1, ctx, x - y1)],
            _ => [with_context!(player0, ctx, x - y0), player1.neg(ctx, &y1)],
        };
        AdditiveTensor { shares }
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
        HostPlacement: PlacementAdd<C, R, R, R>,
        HostPlacement: PlacementMul<C, R, R, R>,
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
        HostPlacement: PlacementMul<C, R, R, R>,
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
        HostPlacement: PlacementMul<C, R, R, R>,
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

#[derive(Clone, Debug, PartialEq)]
pub struct AdditiveMulOp {
    sig: Signature,
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

impl AdditiveMulOp {
    fn ring_add_kernel<C: Context, R>(
        ctx: &C,
        add: &AdditivePlacement,
        x: R,
        y: AdditiveTensor<R>,
    ) -> AdditiveTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementMul<C, R, R, R>,
    {
        let (player0, player1) = add.host_placements();

        let AdditiveTensor { shares: [y0, y1] } = &y;

        let z0 = with_context!(player0, ctx, x * y0);
        let z1 = with_context!(player1, ctx, x * y1);

        AdditiveTensor { shares: [z0, z1] }
    }

    fn add_ring_kernel<C: Context, R>(
        ctx: &C,
        add: &AdditivePlacement,
        x: AdditiveTensor<R>,
        y: R,
    ) -> AdditiveTensor<R>
    where
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementMul<C, R, R, R>,
    {
        let (player0, player1) = add.host_placements();

        let AdditiveTensor { shares: [x0, x1] } = &x;

        let z0 = with_context!(player0, ctx, x0 * y);
        let z1 = with_context!(player1, ctx, x1 * y);

        AdditiveTensor { shares: [z0, z1] }
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
    HostPlacement: PlacementSub<C, R, R, R>,
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
                    let z = self.add(ctx, &x0, &x1);
                    z
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
    AdditivePlacement: PlacementSub<C, AdditiveTensor<R>, AdditiveTensor<R>, AdditiveTensor<R>>,
    AdditivePlacement: PlacementMul<C, AdditiveTensor<R>, R, AdditiveTensor<R>>,
    HostPlacement: PlacementReveal<C, AdditiveTensor<R>, R>,
    HostPlacement: PlacementShl<C, R, R>,
    HostPlacement: PlacementShr<C, R, R>,
    AdditivePlacement: PlacementArithmeticXor<C, R>,
    AdditivePlacement: PlacementShl<C, AdditiveTensor<R>, AdditiveTensor<R>>,
    R: Shl<usize, Output = R> + Into<Value> + Clone,
    HostPlacement: PlacementSample<C, R>,
    AdditivePlacement: PlacementFill<C, Shape, AdditiveTensor<R>>, // TODO: Fix shape; Use type parameter
    HostPlacement: PlacementBitCompose<C, R> + PlacementKeyGen<C, K> + PlacementSub<C, R, R, R>,
    HostPlacement: PlacementOnes<C, Shape, R>,
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

        let k = R::SIZE - 1;
        // TODO(Dragos)this is optional if we work with unsigned numbers
        let x_shape = Shape(vec![1], player_a.clone());

        let ones = player_a.ones(ctx, &x_shape);

        let twok = self.fill(
            ctx,
            player_a.shl(ctx, k, &ones).into(),
            &Shape(vec![1], player_a.clone()),
        );
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
        let remainder = self.fill(ctx, player_a.shl(ctx, k - 1 - m, &ones).into(), &x_shape);
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
        for i in 0..3 {
            let share0 = provider.sample(ctx);
            let share1 = provider.sub(ctx, &tmp[i], &share0);
            // TODO(Dragos) this could probably be optimized by sending the key to p0
            results.push(AdditiveTensor {
                shares: [share0.clone(), share1.clone()],
            })
        }
        (results[0].clone(), results[1].clone(), results[2].clone())
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
    fn kernel<C: Context, R: Clone>(ctx: &C, rep: &ReplicatedPlacement, x: R) -> ReplicatedTensor<R>
    where
        R: Into<C::Value> + TryFrom<C::Value> + 'static,
        R: Placed<Placement = HostPlacement>,
        HostPlacement: PlacementSample<C, R>,
        HostPlacement: PlacementAdd<C, R, R, R>,
        HostPlacement: PlacementSub<C, R, R, R>,
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
    fn kernel<C: Context, R: Clone>(ctx: &C, plc: &HostPlacement, xe: ReplicatedTensor<R>) -> R
    where
        R: Clone + 'static,
        HostPlacement: PlacementAdd<C, R, R, R>,
    {
        let ReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &xe;

        with_context!(plc, ctx, x00 + x10 + x21)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepTruncPrOp {
    sig: Signature,
    amount: usize,
}

modelled!(PlacementTruncPr::trunc_pr, ReplicatedPlacement, attributes[amount: usize] (ReplicatedSetup, Replicated64Tensor) -> Replicated64Tensor, RepTruncPrOp);

kernel! {
    RepTruncPrOp,
    [
        (ReplicatedPlacement,  (ReplicatedSetup, Replicated64Tensor) -> Replicated64Tensor => attributes[amount] Self::kernel),
    ]
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

impl RepTruncPrOp {
    fn kernel<C: Context, R, K>(
        ctx: &C,
        rep: &ReplicatedPlacement,
        amount: usize,
        s: AbstractReplicatedSetup<K>,
        xe: ReplicatedTensor<R>,
    ) -> ReplicatedTensor<R>
    where
        R: Clone + Into<C::Value> + TryFrom<C::Value> + RingSize,
        HostPlacement: PlacementKeyGen<C, K>,
        AdditivePlacement: PlacementTruncPrWithPrep<C, R, K>
            + PlacementRepToAdd<C, ReplicatedTensor<R>, AdditiveTensor<R>>,
    {
        let m = amount;

        let (player0, player1, player2) = rep.host_placements();
        let add_plc = AdditivePlacement {
            players: [player0.player, player1.player],
        };
        let x_add = add_plc.rep_to_add(ctx, &xe);
        let x_trunc = add_plc.trunc_pr(ctx, &x_add, m, player2);

        let signed = true;
        let ReplicatedTensor {
            shares: [[x00, x10], [x11, x21], [x22, x02]],
        } = &xe;

        ReplicatedTensor {
            shares: [
                [x00.clone(), x10.clone()],
                [x11.clone(), x21.clone()],
                [x22.clone(), x02.clone()],
            ],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdditiveRevealOp {
    sig: Signature,
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

impl AdditiveRevealOp {
    fn kernel<C: Context, R: Clone>(ctx: &C, plc: &HostPlacement, xe: AdditiveTensor<R>) -> R
    where
        HostPlacement: PlacementAdd<C, R, R, R>,
    {
        let AdditiveTensor { shares: [x0, x1] } = &xe;
        with_context!(plc, ctx, x1 + x0)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FillOp {
    sig: Signature,
    value: Value,
}

modelled!(PlacementFill::fill, HostPlacement, attributes[value: Value] (Shape) -> Ring64Tensor, FillOp);
modelled!(PlacementFill::fill, HostPlacement, attributes[value: Value] (Shape) -> Ring128Tensor, FillOp);
modelled!(PlacementFill::fill, HostPlacement, attributes[value: Value] (Shape) -> BitTensor, FillOp);
modelled!(PlacementFill::fill, AdditivePlacement, attributes[value: Value] (Shape) -> Additive64Tensor, FillOp);
modelled!(PlacementFill::fill, AdditivePlacement, attributes[value: Value] (Shape) -> Additive128Tensor, FillOp);

kernel! {
    FillOp,
    [
        (HostPlacement, (Shape) -> Ring64Tensor => attributes[value] Self::kernel64),
        (HostPlacement, (Shape) -> Ring128Tensor => attributes[value] Self::kernel128),
        (HostPlacement, (Shape) -> BitTensor => attributes[value] Self::kernel8),
        (AdditivePlacement, (Shape) -> Additive64Tensor => attributes[value] Self::additive_kernel64),
        (AdditivePlacement, (Shape) -> Additive128Tensor => attributes[value] Self::additive_kernel128),
    ]
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
pub struct OnesOp {
    sig: Signature,
}

modelled!(PlacementOnes::ones, HostPlacement, (Shape) -> Ring64Tensor, OnesOp);
modelled!(PlacementOnes::ones, HostPlacement, (Shape) -> Ring128Tensor, OnesOp);

kernel! {
    OnesOp,
    [
        (HostPlacement, (Shape) -> Ring64Tensor => Self::kernel),
        (HostPlacement, (Shape) -> Ring128Tensor => Self::kernel),
    ]
}

impl OnesOp {
    fn kernel<C: Context, T>(ctx: &C, plc: &HostPlacement, shape: &Shape) -> RingTensor<T>
    where
        T: From<bool>,
    {
        // TODO(Dragos) atm we're not using shape
        RingTensor::<T>::fill(T::from(true), plc.clone())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RingAddOp {
    sig: Signature,
}

// NOTE uncomment the next line to see the kernel check system in action
// modelled!(PlacementAdd::add, HostPlacement, (Ring32Tensor, Ring32Tensor) -> Ring32Tensor, RingAddOp);
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
pub struct RingNegOp {
    sig: Signature,
}

modelled!(PlacementNeg::neg, HostPlacement, (Ring64Tensor) -> Ring64Tensor, RingNegOp);
modelled!(PlacementNeg::neg, HostPlacement, (Ring128Tensor) -> Ring128Tensor, RingNegOp);

kernel! {
    RingNegOp,
    [
        (HostPlacement, (Ring64Tensor) -> Ring64Tensor => Self::kernel),
        (HostPlacement, (Ring128Tensor) -> Ring128Tensor => Self::kernel),
    ]
}

impl RingNegOp {
    fn kernel<C: Context, T>(_ctx: &C, _plc: &HostPlacement, x: RingTensor<T>) -> RingTensor<T>
    where
        RingTensor<T>: Neg<Output = RingTensor<T>>,
    {
        -x
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
pub struct RingShrOp {
    sig: Signature,
    amount: usize,
}

modelled!(PlacementShr::shr, HostPlacement, attributes[amount: usize] (Ring64Tensor) -> Ring64Tensor, RingShrOp);
modelled!(PlacementShr::shr, HostPlacement, attributes[amount: usize] (Ring128Tensor) -> Ring128Tensor, RingShrOp);

kernel! {
    RingShrOp,
    [
        (HostPlacement, (Ring64Tensor) -> Ring64Tensor => attributes[amount] Self::kernel),
        (HostPlacement, (Ring128Tensor) -> Ring128Tensor => attributes[amount] Self::kernel),
    ]
}

impl RingShrOp {
    fn kernel<C: Context, T>(
        _ctx: &C,
        _plc: &HostPlacement,
        amount: usize,
        x: RingTensor<T>,
    ) -> RingTensor<T>
    where
        RingTensor<T>: Shr<usize, Output = RingTensor<T>>,
    {
        x >> amount
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
