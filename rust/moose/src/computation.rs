use crate::additive::{AbstractAdditiveTensor, Additive128Tensor, Additive64Tensor};
use crate::bit::BitTensor;
use crate::error::{Error, Result};
use crate::fixedpoint::{Fixed128Tensor, Fixed64Tensor, FixedTensor};
use crate::kernels::Session;
use crate::prim::{Nonce, PrfKey, RawNonce, RawPrfKey, RawSeed, Seed};
use crate::replicated::{
    AbstractReplicatedSetup, AbstractReplicatedShape, AbstractReplicatedTensor,
    Replicated128Tensor, Replicated64Tensor, ReplicatedBitTensor, ReplicatedSetup, ReplicatedShape,
};
use crate::ring::{Ring128Tensor, Ring64Tensor};
use crate::host::{
    HostFloat32Tensor, HostFloat64Tensor, Int16Tensor, Int32Tensor, Int64Tensor, Int8Tensor, RawShape,
    HostShape, HostTensor, Uint16Tensor, Uint32Tensor, Uint64Tensor, Uint8Tensor,
};
use crate::symbolic::{Symbolic, SymbolicSession};
use derive_more::Display;
use macros::ShortName;
use paste::paste;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

pub type RendezvousKey = str;

#[derive(Clone, Debug, Display)]
pub struct SessionId(String);

impl<S: Into<String>> From<S> for SessionId {
    fn from(s: S) -> SessionId {
        SessionId(s.into())
    }
}

impl SessionId {
    pub fn as_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

pub trait KnownType<S: Session> {
    type Type;
    const TY: Ty;
}

// Constants are trivial values. They are what can live on the nodes of the computation graph.
// Constant can not be a Unit, an Unknown or a complex structure such as ReplicatedTensor.
macro_rules! constants {
    ($($val:ident $($t:ident)?,)+) => {

        #[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
        pub enum Constant {
            $($val($val),)+
            // TODO promote below to match other values
            Bit(u8),
            Float32(f32),
            Float64(f64),
            Ring64(u64),
            Ring128(u128),
        }

        impl Constant {
            pub fn ty(&self) -> Ty {
                match self {
                    $(Constant::$val(_) => constants!(@ty $val $($t)?),)+
                    // TODO promote below to match other values
                    Constant::Bit(_) => Ty::Bit,
                    Constant::Float32(_) => Ty::Float32,
                    Constant::Float64(_) => Ty::Float64,
                    Constant::Ring64(_) => Ty::Ring64,
                    Constant::Ring128(_) => Ty::Ring128,
                }
            }

            pub fn place(&self, plc: &HostPlacement) -> Value {
                match self {
                    $(
                        Constant::$val(x) => {constants!(@value(x.clone(), plc.clone().into()) $val $($t)?)},
                    )+
                    // TODO promote below to match other values
                    Constant::Bit(x) => Value::Bit(x.clone()),
                    Constant::Float32(x) => Value::Float32(x.clone()),
                    Constant::Float64(x) => Value::Float64(x.clone()),
                    Constant::Ring64(x) => Value::Ring64(x.clone()),
                    Constant::Ring128(x) => Value::Ring128(x.clone()),
                }
            }
        }

        $(
        impl From<$val> for Constant {
            fn from(x: $val) -> Self {
                Constant::$val(x)
            }
        }
        )+

    };
    (@ty $val:ident $t:ident) => {Ty::$t};
    (@ty $val:ident) => {Ty::$val};

    (@value($x:expr, $plc:expr) $val:ident $t:ident) => {Value::$t($t($x, $plc))};
    (@value($x:expr, $plc:expr) $val:ident) => {Value::$val($x)};
}

// The lines with 2 identifiers are for linking to the "Placed" values - the types whose `Value` incarnation has a placement already.
// The lines with 1 identifier are for linking to the "Unplaced" values, where the Constant and Value are essentially the same and can be converted easily.
constants![
    RawShape HostShape,
    RawSeed Seed,
    RawPrfKey PrfKey,
    RawNonce Nonce,
    String,
    BitTensor,
    Ring64Tensor,
    Ring128Tensor,
    HostFloat32Tensor,
    HostFloat64Tensor,
    Int8Tensor,
    Int16Tensor,
    Int32Tensor,
    Int64Tensor,
    Uint8Tensor,
    Uint16Tensor,
    Uint32Tensor,
    Uint64Tensor,
];

impl From<u8> for Constant {
    // TODO: not obvious that u64 is always Ring64
    fn from(x: u8) -> Self {
        Constant::Bit(x)
    }
}

impl From<u64> for Constant {
    // TODO: not obvious that u64 is always Ring64
    fn from(x: u64) -> Self {
        Constant::Ring64(x)
    }
}

impl From<u128> for Constant {
    // TODO: not obvious that u64 is always Ring64
    fn from(x: u128) -> Self {
        Constant::Ring128(x)
    }
}

// Values are anything that can flow along the edges of the computation graph.
// Some values are just placed constants, but some could be more complex.
macro_rules! values {
    ($(($val:ident, $sym_val:ty),)+) => {

        #[derive(Serialize, Deserialize, PartialEq, Eq, Copy, Clone, Debug, Display)]
        pub enum Ty {
            Unknown,
            $($val,)+
            // TODO promote below to match other values
            Bit,
            Float32,
            Float64,
            Ring64,
            Ring128,
        }

        #[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
        pub enum Value {
            $($val($val),)+
            // TODO promote below to match other values
            Bit(u8),
            Float32(f32),
            Float64(f64),
            Ring64(u64),
            Ring128(u128),
        }

        impl Value {
            pub fn ty(&self) -> Ty {
                match self {
                    $(Value::$val(_) => Ty::$val,)+
                    // TODO promote below to match other values
                    Value::Bit(_) => Ty::Bit,
                    Value::Float32(_) => Ty::Float32,
                    Value::Float64(_) => Ty::Float64,
                    Value::Ring64(_) => Ty::Ring64,
                    Value::Ring128(_) => Ty::Ring128,
                }
            }
        }

        $(
        impl From<$val> for Value {
            fn from(x: $val) -> Self {
                Value::$val(x)
            }
        }
        )+

        $(
        impl From<&$val> for Value {
            fn from(x: &$val) -> Self {
                Value::$val(x.clone())
            }
        }
        )+

        $(
        impl TryFrom<Value> for $val {
            type Error = Error;
            fn try_from(v: Value) -> Result<Self> {
                match v {
                    Value::$val(x) => Ok(x),
                    _ => Err(Error::TypeMismatch {
                        expected: stringify!($val).to_string(),
                        found: v.ty(),
                    }),
                }
            }
        }
        )+

        $(
        impl<'v> TryFrom<&'v Value> for &'v $val {
            type Error = Error;
            fn try_from(v: &'v Value) -> Result<Self> {
                match v {
                    Value::$val(x) => Ok(x),
                    _ => Err(Error::TypeMismatch {
                        expected: stringify!($val).to_string(),
                        found: v.ty(),
                    }),
                }
            }
        }
        )+

        $(
        impl KnownType<crate::kernels::SyncSession> for $val {
            type Type = $val;
            const TY: Ty = Ty::$val;
        }
        )+

        #[derive(PartialEq, Clone, Debug)]
        pub enum SymbolicValue {
            $($val($sym_val),)+
        }

        impl SymbolicValue {
            pub fn ty(&self) -> Ty {
                match self {
                    $(SymbolicValue::$val(_) => Ty::$val,)+
                    // TODO promote below to match other values
                    // SymbolicValue::Unit => Ty::Unit,
                    // SymbolicValue::Bit(_) => Ty::Bit,
                    // SymbolicValue::Float32(_) => Ty::Float32,
                    // SymbolicValue::Float64(_) => Ty::Float64,
                    // SymbolicValue::Ring64(_) => Ty::Ring64,
                    // SymbolicValue::Ring128(_) => Ty::Ring128,
                }
            }
        }

        $(
        impl From<$sym_val> for SymbolicValue {
            fn from(x: $sym_val) -> Self {
                SymbolicValue::$val(x)
            }
        }
        )+

        $(
        impl TryFrom<SymbolicValue> for $sym_val {
            type Error = Error;
            fn try_from(v: SymbolicValue) -> Result<Self> {
                match v {
                    SymbolicValue::$val(x) => Ok(x),
                    _ => Err(Error::TypeMismatch {
                        expected: stringify!($val).to_string(),
                        found: v.ty(),
                    }),
                }
            }
        }
        )+

        $(
        impl KnownType<crate::symbolic::SymbolicSession> for $val {
            type Type = $sym_val;
            const TY: Ty = Ty::$val;
        }
        )+
    };
}

impl From<HostShape> for Symbolic<HostShape> {
    fn from(x: HostShape) -> Self {
        Symbolic::Concrete(x)
    }
}

impl From<Ring64Tensor> for Symbolic<Ring64Tensor> {
    fn from(x: Ring64Tensor) -> Self {
        Symbolic::Concrete(x)
    }
}

impl From<Ring128Tensor> for Symbolic<Ring128Tensor> {
    fn from(x: Ring128Tensor) -> Self {
        Symbolic::Concrete(x)
    }
}

impl From<BitTensor> for Symbolic<BitTensor> {
    fn from(x: BitTensor) -> Self {
        Symbolic::Concrete(x)
    }
}

impl<R> From<AbstractReplicatedTensor<R>> for Symbolic<AbstractReplicatedTensor<R>>
where
    R: Placed<Placement = HostPlacement>,
{
    fn from(x: AbstractReplicatedTensor<R>) -> Self {
        Symbolic::Concrete(x)
    }
}

impl<RingT, RepT> From<FixedTensor<RingT, RepT>> for Symbolic<FixedTensor<RingT, RepT>>
where
    RingT: Placed<Placement = HostPlacement>,
    RepT: Placed<Placement = ReplicatedPlacement>,
{
    fn from(x: FixedTensor<RingT, RepT>) -> Self {
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

impl<S> From<AbstractReplicatedShape<S>> for Symbolic<AbstractReplicatedShape<S>>
where
    S: Placed<Placement = HostPlacement>,
{
    fn from(x: AbstractReplicatedShape<S>) -> Self {
        Symbolic::Concrete(x)
    }
}

impl<R> From<AbstractAdditiveTensor<R>> for Symbolic<AbstractAdditiveTensor<R>>
where
    R: Placed<Placement = HostPlacement>,
{
    fn from(x: AbstractAdditiveTensor<R>) -> Self {
        Symbolic::Concrete(x)
    }
}

impl<T> From<HostTensor<T>> for Symbolic<HostTensor<T>> {
    fn from(x: HostTensor<T>) -> Self {
        Symbolic::Concrete(x)
    }
}

impl<R> TryFrom<Symbolic<AbstractAdditiveTensor<R>>> for AbstractAdditiveTensor<R>
where
    R: Placed<Placement = HostPlacement>,
{
    type Error = Error;
    fn try_from(v: Symbolic<AbstractAdditiveTensor<R>>) -> crate::error::Result<Self> {
        match v {
            Symbolic::Concrete(x) => Ok(x),
            _ => Err(Error::Unexpected), // TODO err message
        }
    }
}

impl<R> TryFrom<Symbolic<AbstractReplicatedTensor<R>>> for AbstractReplicatedTensor<R>
where
    R: Placed<Placement = HostPlacement>,
{
    type Error = Error;
    fn try_from(v: Symbolic<AbstractReplicatedTensor<R>>) -> crate::error::Result<Self> {
        match v {
            Symbolic::Concrete(x) => Ok(x),
            _ => Err(Error::Unexpected), // TODO err message
        }
    }
}

impl<RingT, RepT> TryFrom<Symbolic<FixedTensor<RingT, RepT>>> for FixedTensor<RingT, RepT>
where
    RingT: Placed<Placement = HostPlacement>,
    RepT: Placed<Placement = ReplicatedPlacement>,
{
    type Error = ();
    fn try_from(v: Symbolic<FixedTensor<RingT, RepT>>) -> std::result::Result<Self, ()> {
        match v {
            Symbolic::Concrete(x) => Ok(x),
            _ => Err(()),
        }
    }
}

impl<K> TryFrom<Symbolic<AbstractReplicatedSetup<K>>> for AbstractReplicatedSetup<K>
where
    K: Placed<Placement = HostPlacement>,
{
    type Error = Error;
    fn try_from(v: Symbolic<AbstractReplicatedSetup<K>>) -> crate::error::Result<Self> {
        match v {
            Symbolic::Concrete(x) => Ok(x),
            _ => Err(Error::Unexpected), // TODO err message
        }
    }
}

impl<S> TryFrom<Symbolic<AbstractReplicatedShape<S>>> for AbstractReplicatedShape<S>
where
    S: Placed<Placement = HostPlacement>,
{
    type Error = Error;
    fn try_from(v: Symbolic<AbstractReplicatedShape<S>>) -> crate::error::Result<Self> {
        match v {
            Symbolic::Concrete(x) => Ok(x),
            _ => Err(Error::Unexpected), // TODO err message
        }
    }
}

impl TryFrom<Symbolic<BitTensor>> for BitTensor {
    type Error = Error;
    fn try_from(v: Symbolic<BitTensor>) -> crate::error::Result<Self> {
        match v {
            Symbolic::Concrete(x) => Ok(x),
            _ => Err(Error::Unexpected), // TODO err message
        }
    }
}

impl<T> TryFrom<Symbolic<HostTensor<T>>> for HostTensor<T> {
    type Error = Error;
    fn try_from(v: Symbolic<HostTensor<T>>) -> crate::error::Result<Self> {
        match v {
            Symbolic::Concrete(x) => Ok(x),
            _ => Err(Error::Unexpected), // TODO err message
        }
    }
}

values![
    (Unit, Symbolic<Unit>),
    (HostShape, Symbolic<HostShape>),
    (Seed, Symbolic<Seed>),
    (PrfKey, Symbolic<PrfKey>),
    (Nonce, Symbolic<Nonce>),
    (String, Symbolic<String>),
    (BitTensor, Symbolic<BitTensor>),
    (Ring64Tensor, Symbolic<Ring64Tensor>),
    (Ring128Tensor, Symbolic<Ring128Tensor>),
    (HostFloat32Tensor, Symbolic<HostFloat32Tensor>),
    (HostFloat64Tensor, Symbolic<HostFloat64Tensor>),
    (Int8Tensor, Symbolic<Int8Tensor>),
    (Int16Tensor, Symbolic<Int16Tensor>),
    (Int32Tensor, Symbolic<Int32Tensor>),
    (Int64Tensor, Symbolic<Int64Tensor>),
    (Uint8Tensor, Symbolic<Uint8Tensor>),
    (Uint16Tensor, Symbolic<Uint16Tensor>),
    (Uint32Tensor, Symbolic<Uint32Tensor>),
    (Uint64Tensor, Symbolic<Uint64Tensor>),
    (
        Fixed64Tensor,
        Symbolic<
            FixedTensor<
                <Ring64Tensor as KnownType<SymbolicSession>>::Type,
                <Replicated64Tensor as KnownType<SymbolicSession>>::Type,
            >,
        >
    ),
    (
        Fixed128Tensor,
        Symbolic<
            FixedTensor<
                <Ring128Tensor as KnownType<SymbolicSession>>::Type,
                <Replicated128Tensor as KnownType<SymbolicSession>>::Type,
            >,
        >
    ),
    (
        Replicated64Tensor,
        Symbolic<AbstractReplicatedTensor<<Ring64Tensor as KnownType<SymbolicSession>>::Type>>
    ),
    (
        Replicated128Tensor,
        Symbolic<AbstractReplicatedTensor<<Ring128Tensor as KnownType<SymbolicSession>>::Type>>
    ),
    (
        ReplicatedBitTensor,
        Symbolic<AbstractReplicatedTensor<<BitTensor as KnownType<SymbolicSession>>::Type>>
    ),
    (
        ReplicatedSetup,
        Symbolic<AbstractReplicatedSetup<<PrfKey as KnownType<SymbolicSession>>::Type>>
    ),
    (
        ReplicatedShape,
        Symbolic<AbstractReplicatedShape<<HostShape as KnownType<SymbolicSession>>::Type>>
    ),
    (
        Additive64Tensor,
        Symbolic<AbstractAdditiveTensor<<Ring64Tensor as KnownType<SymbolicSession>>::Type>>
    ),
    (
        Additive128Tensor,
        Symbolic<AbstractAdditiveTensor<<Ring128Tensor as KnownType<SymbolicSession>>::Type>>
    ),
];

// Unit is still special. Placed unit is just a host placement.
#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct Unit(pub HostPlacement);

impl Placed for Unit {
    type Placement = HostPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        Ok(self.0.clone())
    }
}

#[derive(Serialize, Deserialize, PartialEq, Copy, Clone, Debug)]
pub enum Signature {
    Nullary(NullarySignature),
    Unary(UnarySignature),
    Binary(BinarySignature),
    Ternary(TernarySignature),
}

#[derive(Serialize, Deserialize, PartialEq, Copy, Clone, Debug)]
pub struct NullarySignature {
    pub ret: Ty,
}

#[derive(Serialize, Deserialize, PartialEq, Copy, Clone, Debug)]
pub struct UnarySignature {
    pub arg0: Ty,
    pub ret: Ty,
}

#[derive(Serialize, Deserialize, PartialEq, Copy, Clone, Debug)]
pub struct BinarySignature {
    pub arg0: Ty,
    pub arg1: Ty,
    pub ret: Ty,
}

#[derive(Serialize, Deserialize, PartialEq, Copy, Clone, Debug)]
pub struct TernarySignature {
    pub arg0: Ty,
    pub arg1: Ty,
    pub arg2: Ty,
    pub ret: Ty,
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

impl Signature {
    pub fn nullary(ret: Ty) -> Signature {
        NullarySignature { ret }.into()
    }
    pub fn unary(arg0: Ty, ret: Ty) -> Signature {
        UnarySignature { arg0, ret }.into()
    }
    pub fn binary(arg0: Ty, arg1: Ty, ret: Ty) -> Signature {
        BinarySignature { arg0, arg1, ret }.into()
    }
    pub fn ternary(arg0: Ty, arg1: Ty, arg2: Ty, ret: Ty) -> Signature {
        TernarySignature {
            arg0,
            arg1,
            arg2,
            ret,
        }
        .into()
    }
    pub fn ret(&self) -> Ty {
        match self {
            Signature::Nullary(s) => s.ret,
            Signature::Unary(s) => s.ret,
            Signature::Binary(s) => s.ret,
            Signature::Ternary(s) => s.ret,
        }
    }

    pub fn arg(&self, arg: usize) -> Result<Ty> {
        match (self, arg) {
            (Signature::Unary(s), 0) => Ok(s.arg0),
            (Signature::Binary(s), 0) => Ok(s.arg0),
            (Signature::Binary(s), 1) => Ok(s.arg1),
            (Signature::Ternary(s), 0) => Ok(s.arg0),
            (Signature::Ternary(s), 1) => Ok(s.arg1),
            (Signature::Ternary(s), 2) => Ok(s.arg2),
            _ => Err(Error::OperandUnavailable),
        }
    }

    pub fn arity(&self) -> usize {
        match self {
            Signature::Nullary(_) => 0,
            Signature::Unary(_) => 1,
            Signature::Binary(_) => 2,
            Signature::Ternary(_) => 3,
        }
    }

    pub fn merge(&mut self, another: Signature) -> anyhow::Result<()> {
        match (self, &another) {
            (Signature::Nullary(s), Signature::Nullary(o)) => s.merge(o),
            (Signature::Unary(s), Signature::Unary(o)) => s.merge(o),
            (Signature::Binary(s), Signature::Binary(o)) => s.merge(o),
            (Signature::Ternary(s), Signature::Ternary(o)) => s.merge(o),
            (Signature::Nullary(s), o) => Err(anyhow::anyhow!(
                "Can not merge {:?} with an incompatible signature {:?}",
                s,
                o
            )),
            (Signature::Unary(s), o) => Err(anyhow::anyhow!(
                "Can not merge {:?} with an incompatible signature {:?}",
                s,
                o
            )),
            (Signature::Binary(s), o) => Err(anyhow::anyhow!(
                "Can not merge {:?} with an incompatible signature {:?}",
                s,
                o
            )),
            (Signature::Ternary(s), o) => Err(anyhow::anyhow!(
                "Can not merge {:?} with an incompatible signature {:?}",
                s,
                o
            )),
        }
    }
}

impl NullarySignature {
    pub fn merge(&mut self, another: &NullarySignature) -> anyhow::Result<()> {
        if let Some(new_type) = self.ret.merge(&another.ret) {
            self.ret = new_type;
        }
        Ok(())
    }
}

impl UnarySignature {
    pub fn merge(&mut self, another: &UnarySignature) -> anyhow::Result<()> {
        if let Some(new_type) = self.arg0.merge(&another.arg0) {
            self.arg0 = new_type;
        }
        if let Some(new_type) = self.ret.merge(&another.ret) {
            self.ret = new_type;
        }
        Ok(())
    }
}

impl BinarySignature {
    pub fn merge(&mut self, another: &BinarySignature) -> anyhow::Result<()> {
        if let Some(new_type) = self.arg0.merge(&another.arg0) {
            self.arg0 = new_type;
        }
        if let Some(new_type) = self.arg1.merge(&another.arg1) {
            self.arg1 = new_type;
        }
        if let Some(new_type) = self.ret.merge(&another.ret) {
            self.ret = new_type;
        }
        Ok(())
    }
}

impl TernarySignature {
    pub fn merge(&mut self, another: &TernarySignature) -> anyhow::Result<()> {
        if let Some(new_type) = self.arg0.merge(&another.arg0) {
            self.arg0 = new_type;
        }
        if let Some(new_type) = self.arg1.merge(&another.arg1) {
            self.arg1 = new_type;
        }
        if let Some(new_type) = self.arg2.merge(&another.arg2) {
            self.arg2 = new_type;
        }
        if let Some(new_type) = self.ret.merge(&another.ret) {
            self.ret = new_type;
        }
        Ok(())
    }
}

impl Ty {
    /// Merge type information.
    ///
    /// Returns `Some(new_type)` if a merge produced a new type.
    /// Otherwise returns None
    pub fn merge(&self, another: &Ty) -> Option<Ty> {
        match self {
            Ty::Unknown => Some(*another),
            _ => None,
        }
    }
}

macro_rules! operators {
    ($($t:ident,)+) => {

        paste! {
            #[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
            pub enum Operator {
                $($t([<$t Op>]),)+
            }
        }

        $(
        paste! {
            impl From<[<$t Op>]> for Operator {
                fn from(x: [<$t Op>]) -> Operator {
                    Operator::$t(x)
                }
            }
        }
        )+

        impl Operator {
            pub fn sig(&self) -> &Signature {
                match self {
                    $(Operator::$t(op) => &op.sig,)+
                }
            }

            pub fn sig_mut(&mut self) -> &mut Signature {
                match self {
                    $(Operator::$t(op) => &mut op.sig,)+
                }
            }

            pub fn short_name(&self) -> &str {
                match self {
                    $(Operator::$t(op) => op.short_name(),)+
                }
            }
        }
    }
}

operators![
    Identity,
    Load,
    Save,
    Send,
    Receive,
    Input,
    Output,
    Constant,
    Shape,
    BitFill,
    RingFill,
    AdtFill,
    RepFill,
    StdAdd,
    StdSub,
    StdMul,
    StdDiv,
    StdDot,
    StdMean,
    StdExpandDims,
    StdReshape,
    StdAtLeast2D,
    StdSlice,
    StdSum,
    StdOnes,
    StdConcatenate,
    StdTranspose,
    StdInverse,
    RingAdd,
    RingSub,
    RingNeg,
    RingMul,
    RingDot,
    RingSum,
    RingSample,
    RingShl,
    RingShr,
    RingInject,
    RingToBit,
    BitExtract,
    BitSample,
    BitXor,
    BitAnd,
    PrimDeriveSeed,
    PrimPrfKeyGen,
    FixedpointEncode,
    FixedpointDecode,
    FixedpointAdd,
    FixedpointSub,
    FixedpointMul,
    FixedpointDot,
    FixedpointTruncPr,
    FixedpointMean,
    FixedpointSum,
    FixedpointRingEncode,
    FixedpointRingDecode,
    FixedpointRingMean,
    AdtReveal,
    AdtAdd,
    AdtSub,
    AdtMul,
    AdtShl,
    AdtToRep,
    RepAbs,
    RepSetup,
    RepShare,
    RepReveal,
    RepAdd,
    RepSub,
    RepMul,
    RepMsb,
    RepDot,
    RepMean,
    RepSum,
    RepTruncPr,
    RepToAdt,
];

pub trait HasShortName {
    fn short_name(&self) -> &str;
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct SendOp {
    pub sig: Signature,
    pub rendezvous_key: String,
    pub receiver: Role,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct IdentityOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct ReceiveOp {
    pub sig: Signature,
    pub rendezvous_key: String,
    pub sender: Role,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct InputOp {
    pub sig: Signature,
    pub arg_name: String,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct OutputOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct LoadOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct SaveOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct ConstantOp {
    pub sig: Signature,
    pub value: Constant, // TODO Box<Constant> or Box inside Constant?
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct StdAddOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct StdSubOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct StdMulOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct StdDivOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct StdDotOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct StdMeanOp {
    pub sig: Signature,
    pub axis: Option<u32>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct StdOnesOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct StdConcatenateOp {
    pub sig: Signature,
    pub axis: u32,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct StdAtLeast2DOp {
    pub sig: Signature,
    pub to_column_vector: bool,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct StdExpandDimsOp {
    pub sig: Signature,
    pub axis: Vec<u32>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct StdReshapeOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct StdSliceOp {
    pub sig: Signature,
    pub start: u32,
    pub end: u32,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct StdSumOp {
    pub sig: Signature,
    pub axis: Option<u32>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct StdTransposeOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct StdInverseOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct ShapeOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct BitFillOp {
    pub sig: Signature,
    pub value: Constant,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RingToBitOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RingFillOp {
    pub sig: Signature,
    pub value: Constant,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct AdtFillOp {
    pub sig: Signature,
    pub value: Constant,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct PrimDeriveSeedOp {
    pub sig: Signature,
    pub sync_key: RawNonce,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct PrimPrfKeyGenOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RingAddOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RingSubOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RingNegOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RingMulOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RingDotOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RingSumOp {
    pub sig: Signature,
    pub axis: Option<u32>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RingSampleOp {
    pub sig: Signature,
    pub max_value: Option<u64>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RingShlOp {
    pub sig: Signature,
    pub amount: usize,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RingShrOp {
    pub sig: Signature,
    pub amount: usize,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RingInjectOp {
    pub sig: Signature,
    pub bit_idx: usize,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct BitExtractOp {
    pub sig: Signature,
    pub bit_idx: usize,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct BitSampleOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct BitXorOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct BitAndOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FixedpointEncodeOp {
    pub sig: Signature,
    pub precision: u32,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FixedpointDecodeOp {
    pub sig: Signature,
    pub precision: u32,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FixedpointAddOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FixedpointSubOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FixedpointMulOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FixedpointDotOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FixedpointTruncPrOp {
    pub sig: Signature,
    pub precision: u32,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FixedpointMeanOp {
    pub sig: Signature,
    pub axis: Option<u32>,
    pub scaling_base: u64,
    pub scaling_exp: u32,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FixedpointSumOp {
    pub sig: Signature,
    pub axis: Option<u32>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FixedpointRingEncodeOp {
    pub sig: Signature,
    pub scaling_base: u64,
    pub scaling_exp: u32,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FixedpointRingDecodeOp {
    pub sig: Signature,
    pub scaling_base: u64,
    pub scaling_exp: u32,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FixedpointRingMeanOp {
    pub sig: Signature,
    pub axis: Option<u32>,
    pub scaling_base: u64,
    pub scaling_exp: u32,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct AdtRevealOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct AdtAddOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct AdtSubOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct AdtMulOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct AdtShlOp {
    pub sig: Signature,
    pub amount: usize,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct AdtToRepOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RepAbsOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RepMsbOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RepSetupOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RepShareOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RepRevealOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RepAddOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RepSubOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RepMulOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RepDotOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RepMeanOp {
    pub sig: Signature,
    pub axis: Option<u32>,
    pub precision: u64,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RepSumOp {
    pub sig: Signature,
    pub axis: Option<u32>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RepTruncPrOp {
    pub sig: Signature,
    pub amount: u32,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RepToAdtOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RepFillOp {
    pub sig: Signature,
    pub value: Constant,
}

pub trait KnownPlacement {
    const TY: PlacementTy;

    fn ty(&self) -> PlacementTy {
        Self::TY
    }
}

macro_rules! placements {
    ($($p:ident,)+) => {
        paste! {
            #[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
            pub enum Placement {
                $($p([<$p Placement>]),)+
            }
        }

        paste! {
            #[derive(Serialize, Deserialize, PartialEq, Eq, Clone, Debug)]
            pub enum PlacementTy {
                $($p,)+
            }
        }

        impl Placement {
            pub fn ty(&self) -> PlacementTy {
                match self {
                    $(Placement::$p(plc) => plc.ty(),)+
                }
            }
        }

        paste! {
            $(
            impl From<[<$p Placement>]> for Placement {
                fn from(x: [<$p Placement>]) -> Placement {
                    Placement::$p(x)
                }
            }
            )+
        }

        paste! {
            $(
            impl From<&[<$p Placement>]> for Placement {
                fn from(x: &[<$p Placement>]) -> Placement {
                    Placement::$p(x.clone())
                }
            }
            )+
        }

        paste! {
            $(
            impl TryFrom<Placement> for [<$p Placement>] {
                type Error = Error;

                fn try_from(x: Placement) -> Result<Self> {
                    match x {
                        Placement::$p(x) => Ok(x),
                        _ => Err(Error::OperandUnavailable),
                    }
                }
            }
            )+
        }

        paste! {
            $(
            impl KnownPlacement for [<$p Placement>] {
                const TY: PlacementTy = PlacementTy::$p;
            }
            )+
        }
    };
}

placements![Host, Replicated, Additive,];

#[derive(Serialize, Deserialize, Display, Clone, Debug, Hash, Eq, PartialEq)]
pub struct Role(pub String);

impl From<String> for Role {
    fn from(s: String) -> Self {
        Role(s)
    }
}

impl From<&String> for Role {
    fn from(s: &String) -> Self {
        Role(s.clone())
    }
}

impl From<&str> for Role {
    fn from(s: &str) -> Self {
        Role(s.to_string())
    }
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct HostPlacement {
    pub owner: Role,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug)]
pub struct ReplicatedPlacement {
    pub owners: [Role; 3],
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct AdditivePlacement {
    pub owners: [Role; 2],
}

impl ReplicatedPlacement {
    pub fn host_placements(&self) -> (HostPlacement, HostPlacement, HostPlacement) {
        let player0 = HostPlacement {
            owner: self.owners[0].clone(),
        };
        let player1 = HostPlacement {
            owner: self.owners[1].clone(),
        };
        let player2 = HostPlacement {
            owner: self.owners[2].clone(),
        };
        (player0, player1, player2)
    }
}

impl AdditivePlacement {
    pub fn host_placements(&self) -> (HostPlacement, HostPlacement) {
        let player0 = HostPlacement {
            owner: self.owners[0].clone(),
        };
        let player1 = HostPlacement {
            owner: self.owners[1].clone(),
        };
        (player0, player1)
    }
}

pub trait Placed {
    type Placement;

    fn placement(&self) -> std::result::Result<Self::Placement, Error>;
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Operation {
    pub name: String,
    pub kind: Operator,
    pub inputs: Vec<String>, // TODO(Morten) use indices instead of strings?
    pub placement: Placement,
}

#[derive(Debug)]
pub struct Computation {
    // pub constants: Vec<Value>,
    // pub operators: Vec<Operator>,
    pub operations: Vec<Operation>,
}
