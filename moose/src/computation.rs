//! Moose computation format.

use crate::additive::*;
use crate::error::{Error, Result};
#[cfg(feature = "compile")]
use crate::execution::symbolic::Symbolic;
use crate::execution::Session;
use crate::host::*;
use crate::logical::{TensorDType, TensorShape};
use crate::mirrored::Mirrored3Placement;
use crate::replicated::*;
use crate::textual::ToTextual;
use crate::types::*;
use byteorder::{ByteOrder, LittleEndian};
use derive_more::Display;
use moose_macros::{FromTextual, ShortName, ToTextual};
use paste::paste;
use petgraph::graph::NodeIndex;
use petgraph::Graph;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::fs::File;
use std::fs::OpenOptions;
use std::hash::{Hash, Hasher};
use std::io::prelude::*;
use std::io::BufWriter;
use std::path::Path;

pub const TAG_BYTES: usize = 128 / 8;
static_assertions::const_assert!(TAG_BYTES <= blake3::OUT_LEN);
// TODO: the displayed representation of the RendezvousKey does not match with
// the input. Might need to do something similar to what we did with the
// session id, and have a secure and a logical form of it?
#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, Eq, Hash)]
pub struct RendezvousKey(pub(crate) [u8; TAG_BYTES]);

impl std::fmt::Display for RendezvousKey {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for byte in self.0 {
            write!(f, "{:02X}", byte)?
        }
        Ok(())
    }
}

impl From<u128> for RendezvousKey {
    fn from(v: u128) -> RendezvousKey {
        let mut raw = [0; TAG_BYTES];
        LittleEndian::write_u128(&mut raw, v);
        RendezvousKey(raw)
    }
}

impl TryFrom<String> for RendezvousKey {
    type Error = Error;
    fn try_from(s: String) -> Result<RendezvousKey> {
        Self::try_from(s.as_str())
    }
}

impl TryFrom<&str> for RendezvousKey {
    type Error = Error;
    fn try_from(s: &str) -> Result<RendezvousKey> {
        let s_bytes = s.as_bytes();
        if s_bytes.len() > TAG_BYTES {
            return Err(Error::Unexpected(None)); // TODO more helpful error message
        }
        let mut raw: [u8; TAG_BYTES] = [0; TAG_BYTES];
        for (idx, byte) in s_bytes.iter().enumerate() {
            raw[idx] = *byte;
        }
        Ok(RendezvousKey(raw))
    }
}

impl RendezvousKey {
    pub fn from_bytes(bytes: [u8; TAG_BYTES]) -> Self {
        RendezvousKey(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; TAG_BYTES] {
        &self.0
    }

    pub fn random() -> Self {
        use rand::RngCore;
        let mut raw = [0; TAG_BYTES];
        let mut rng = rand::thread_rng();
        rng.fill_bytes(&mut raw);
        RendezvousKey(raw)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SessionId {
    logical: String,
    secure: [u8; TAG_BYTES],
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.logical)?;
        Ok(())
    }
}

impl TryFrom<&str> for SessionId {
    type Error = Error;
    fn try_from(s: &str) -> Result<SessionId> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(s.as_bytes());
        let mut digest = hasher.finalize_xof();

        let mut raw_hash = [0u8; TAG_BYTES];
        digest.fill(&mut raw_hash);

        let sid = SessionId {
            logical: s.to_string(),
            secure: raw_hash,
        };
        Ok(sid)
    }
}

impl SessionId {
    pub fn as_bytes(&self) -> &[u8; TAG_BYTES] {
        &self.secure
    }

    pub fn random() -> Self {
        use rand::RngCore;
        let mut raw = [0; TAG_BYTES];
        let mut rng = rand::thread_rng();
        rng.fill_bytes(&mut raw);

        let hex_vec: Vec<String> = raw.iter().map(|byte| format!("{:02X}", byte)).collect();
        let hex_string = hex_vec.join("");
        SessionId {
            logical: hex_string,
            secure: raw,
        }
    }
}

/// Type map used to compute the symbolic version of a Moose type.
///
/// Note that this trait is typically not implemented directly, but
/// rather through an implementation of the PartiallySymbolicType map.
#[cfg(feature = "compile")]
pub trait SymbolicType {
    type Type;
}

#[cfg(feature = "compile")]
impl<T> SymbolicType for T
where
    T: PartiallySymbolicType,
    <T as PartiallySymbolicType>::Type: Placed,
{
    type Type = Symbolic<<T as PartiallySymbolicType>::Type>;
}

/// Type map used to compute the almost symbolic version of a Moose type.
///
/// Concretely, this map computes the symbolic version, except for the top-most
/// type. As an example, RepTensor<Symbolic<HostTensor>> is partially symbolic
/// as opposed to the (fully) symbolic type Symbolic<RepTensor<Symbolic<HostTensor>.
#[cfg(feature = "compile")]
pub trait PartiallySymbolicType {
    type Type;
}

/// Type map used to compute the (fully) concrete version of a Moose type.
///
/// For example, Symbolic<RepTensor<Symbolic<HostTensor>>> and RepTensor<HostTensor>
/// are both mapped to RepTensor<HostTensor>.
pub trait CanonicalType {
    type Type;
}

pub trait KnownType<S: Session> {
    type Type;
    const TY: Ty;
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct FixedpointConstant {
    pub value: f64,
    pub precision: usize,
}

pub(crate) trait AsFixedpoint {
    fn as_fixedpoint(&self, precision: usize) -> FixedpointConstant;
}

impl AsFixedpoint for f64 {
    fn as_fixedpoint(&self, precision: usize) -> FixedpointConstant {
        FixedpointConstant {
            value: *self,
            precision,
        }
    }
}

// Constants are trivial values. They are what can live on the nodes of the computation graph.
// Constant can not be a HostUnit, an Unknown or a complex structure such as ReplicatedTensor.
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
            Fixed(FixedpointConstant),
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
                    Constant::Fixed(_) => Ty::Fixed,
                }
            }

            pub fn place(&self, plc: &HostPlacement) -> Value {
                match self {
                    $(
                        Constant::$val(x) => {
                            constants!(@value(x.clone(), plc) $val $(as $t)?)
                        },
                    )+
                    // TODO promote below to match other values
                    Constant::Bit(x) => Value::Bit(Box::new(x.clone())),
                    Constant::Float32(x) => Value::Float32(Box::new(x.clone())),
                    Constant::Float64(x) => Value::Float64(Box::new(x.clone())),
                    Constant::Ring64(x) => Value::Ring64(Box::new(x.clone())),
                    Constant::Ring128(x) => Value::Ring128(Box::new(x.clone())),
                    Constant::Fixed(x) => Value::Fixed(Box::new(x.clone())),
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

    (@value($x:expr, $plc:expr) $val:ident as $t:ident) => {Value::$t(Box::new($t($x, $plc.clone().into())))};
    (@value($x:expr, $plc:expr) $val:ident) => {Value::$val(Box::new($val::place($plc, $x.0)))};
}

// The lines with 2 identifiers are for linking to the "Placed" values - the types whose `Value` incarnation has a placement already.
// The lines with 1 identifier are for linking to the "Unplaced" values, where the Constant and Value are essentially the same and can be converted easily.
constants![
    RawShape HostShape,
    RawSeed HostSeed,
    RawPrfKey HostPrfKey,
    String HostString,
    HostBitTensor,
    HostRing64Tensor,
    HostRing128Tensor,
    HostFloat32Tensor,
    HostFloat64Tensor,
    HostInt8Tensor,
    HostInt16Tensor,
    HostInt32Tensor,
    HostInt64Tensor,
    HostUint8Tensor,
    HostUint16Tensor,
    HostUint32Tensor,
    HostUint64Tensor,
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
impl From<FixedpointConstant> for Constant {
    fn from(x: FixedpointConstant) -> Self {
        Constant::Fixed(FixedpointConstant {
            value: x.value,
            precision: x.precision,
        })
    }
}

macro_rules! anything_to_underscore {
    ($($_:tt)*) => {
        _
    };
}

// Values are anything that can flow along the edges of the computation graph.
// Some values are just placed constants, but some could be more complex.
macro_rules! values {
    ($($val:ident$(($inner:ident::$default:ident))?,)+) => {

        #[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Copy, Clone, Debug, Display)]
        pub enum Ty {
            Unknown,
            $($val$(($inner))?,)+
            // TODO promote below to match other values
            Bit,
            Float32,
            Float64,
            Ring64,
            Ring128,
            Fixed,
        }

        impl Ty {
            pub fn from_name(name: &str, inner: Option<&str>) -> anyhow::Result<Self>
            {
                use std::convert::TryInto;
                match name {
                    "Unknown" => Ok(Ty::Unknown),
                    $(stringify!($val) => Ok(Ty::$val
                        // Optional inner type conversion clause
                        $((
                            TryInto::<$inner>::try_into(
                                inner.ok_or_else(|| anyhow::anyhow!("Expected an inner type for {}", name))?
                            )?
                        ))?
                    ),)+
                    "Bit" => Ok(Ty::Bit),
                    "Float32" => Ok(Ty::Float32),
                    "Float64" => Ok(Ty::Float64),
                    "Ring64" => Ok(Ty::Ring64),
                    "Ring128" => Ok(Ty::Ring128),
                    "Fixed" => Ok(Ty::Fixed),
                    // The names below are deprecated aliases, maintained for a long period of time for compatibility
                    "Seed" => Ok(Ty::HostSeed), // pre v0.1.5
                    "PrfKey" => Ok(Ty::HostPrfKey), // pre v0.1.5
                    "Unit" => Ok(Ty::HostUnit), // pre v0.1.5
                    _ => Err(anyhow::anyhow!("Unsupported type name {}", name)),
                }
            }
        }

        impl HasShortName for Ty {
            fn short_name(&self) -> &str {
                match self {
                    Ty::Unknown => "Unknown",
                    $(Ty::$val$((anything_to_underscore!{$inner}))? => stringify!($val),)+ // TODO: Should we output `inner` in any form?
                    Ty::Bit => "Bit",
                    Ty::Float32 => "Float32",
                    Ty::Float64 => "Float64",
                    Ty::Ring64 => "Ring64",
                    Ty::Ring128 => "Ring128",
                    Ty::Fixed => "Fixed",
                }
            }
        }

        #[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
        pub enum Value {
            $($val(Box<$val>),)+
            // TODO promote below to match other values
            Bit(Box<u8>),
            Float32(Box<f32>),
            Float64(Box<f64>),
            Ring64(Box<u64>),
            Ring128(Box<u128>),
            Fixed(Box<FixedpointConstant>),
        }

        impl Value {
            pub fn ty(&self) -> Ty {
                match self {
                    $(Value::$val(_) => Ty::$val$(($inner::$default))?,)+
                    // TODO promote below to match other values
                    Value::Bit(_) => Ty::Bit,
                    Value::Float32(_) => Ty::Float32,
                    Value::Float64(_) => Ty::Float64,
                    Value::Ring64(_) => Ty::Ring64,
                    Value::Ring128(_) => Ty::Ring128,
                    Value::Fixed(_) => Ty::Fixed,
                }
            }
        }

        $(
        impl From<$val> for Value {
            fn from(x: $val) -> Self {
                Value::$val(Box::new(x))
            }
        }
        )+

        $(
        impl From<&$val> for Value {
            fn from(x: &$val) -> Self {
                Value::$val(Box::new(x.clone()))
            }
        }
        )+

        $(
        impl TryFrom<Value> for $val {
            type Error = Error;
            fn try_from(v: Value) -> Result<Self> {
                match v {
                    Value::$val(x) => Ok(*x),
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
            #[cfg(feature = "sync_execute")]
            impl KnownType<crate::execution::SyncSession> for $val {
                type Type = $val;
                const TY: Ty = Ty::$val$(($inner::$default))?;
            }
        )+

        #[cfg(feature = "compile")]
        #[derive(PartialEq, Clone, Debug)]
        #[allow(clippy::large_enum_variant)]
        pub enum SymbolicValue {
            $($val(Box<<$val as SymbolicType>::Type>),)+
        }

        #[cfg(feature = "compile")]
        impl SymbolicValue {
            pub fn ty(&self) -> Ty {
                match self {
                    $(SymbolicValue::$val(_) => Ty::$val$(($inner::$default))?,)+
                }
            }
        }

        $(
            #[cfg(feature = "compile")]
            impl From<<$val as SymbolicType>::Type> for SymbolicValue {
                fn from(x: <$val as SymbolicType>::Type) -> Self {
                    SymbolicValue::$val(Box::new(x))
                }
            }
        )+

        $(
            #[cfg(feature = "compile")]
            impl TryFrom<SymbolicValue> for <$val as SymbolicType>::Type {
                type Error = Error;
                fn try_from(v: SymbolicValue) -> Result<Self> {
                    match v {
                        SymbolicValue::$val(x) => Ok(*x),
                        _ => Err(Error::TypeMismatch {
                            expected: stringify!($val).to_string(),
                            found: v.ty(),
                        }),
                    }
                }
            }
        )+

        $(
            #[cfg(feature = "compile")]
            impl KnownType<crate::execution::SymbolicSession> for $val {
                type Type = <$val as SymbolicType>::Type;
                const TY: Ty = Ty::$val$(($inner::$default))?;
            }
        )+

        $(
            #[cfg(feature = "async_execute")]
            impl KnownType<crate::execution::AsyncSession> for $val {
                type Type = $val;
                const TY: Ty = Ty::$val$(($inner::$default))?;
            }
        )+

    };
}

values![
    HostUnit,
    HostShape,
    HostSeed,
    HostPrfKey,
    HostString,
    Shape(TensorShape::Unknown),
    Tensor(TensorDType::Unknown),
    HostBitTensor,
    HostBitArray64,
    HostBitArray128,
    HostBitArray224,
    HostBitArray256,
    HostRing64Tensor,
    HostRing128Tensor,
    HostFixed64Tensor,
    HostFixed128Tensor,
    HostFloat32Tensor,
    HostFloat64Tensor,
    HostInt8Tensor,
    HostInt16Tensor,
    HostInt32Tensor,
    HostInt64Tensor,
    HostUint8Tensor,
    HostUint16Tensor,
    HostUint32Tensor,
    HostUint64Tensor,
    HostFixed128AesTensor,
    HostAesKey,
    BooleanTensor,
    Fixed64Tensor,
    Fixed128Tensor,
    Float32Tensor,
    Float64Tensor,
    Uint64Tensor,
    ReplicatedRing64Tensor,
    ReplicatedRing128Tensor,
    ReplicatedBitTensor,
    ReplicatedBitArray64,
    ReplicatedBitArray128,
    ReplicatedBitArray224,
    ReplicatedFixed64Tensor,
    ReplicatedFixed128Tensor,
    ReplicatedUint64Tensor,
    ReplicatedAesKey,
    ReplicatedShape,
    Mirrored3Ring64Tensor,
    Mirrored3Ring128Tensor,
    Mirrored3BitTensor,
    Mirrored3Fixed64Tensor,
    Mirrored3Fixed128Tensor,
    Mirrored3Float32,
    Mirrored3Float64,
    AdditiveBitTensor,
    AdditiveRing64Tensor,
    AdditiveRing128Tensor,
    AdditiveShape,
    Fixed128AesTensor,
    AesKey,
    AesTensor,
];

// HostUnit is still special. Placed unit is just a host placement.
#[derive(Serialize, Deserialize, PartialEq, Eq, Clone, Debug)]
pub struct HostUnit(pub HostPlacement);

#[cfg(feature = "compile")]
impl PartiallySymbolicType for HostUnit {
    type Type = HostUnit;
}

impl Placed for HostUnit {
    type Placement = HostPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        Ok(self.0.clone())
    }
}

impl Ty {
    pub(crate) fn flatten(&self) -> Ty {
        match self {
            Ty::Tensor(_) => Ty::Tensor(TensorDType::Unknown),
            Ty::Shape(_) => Ty::Shape(TensorShape::Unknown),
            _ => *self,
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub enum Signature {
    Nullary(NullarySignature),
    Unary(UnarySignature),
    Binary(BinarySignature),
    Ternary(TernarySignature),
    Variadic(VariadicSignature),
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub struct NullarySignature {
    pub ret: Ty,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub struct UnarySignature {
    pub arg0: Ty,
    pub ret: Ty,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub struct BinarySignature {
    pub arg0: Ty,
    pub arg1: Ty,
    pub ret: Ty,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub struct TernarySignature {
    pub arg0: Ty,
    pub arg1: Ty,
    pub arg2: Ty,
    pub ret: Ty,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub struct VariadicSignature {
    pub args: Ty,
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

impl From<VariadicSignature> for Signature {
    fn from(s: VariadicSignature) -> Signature {
        Signature::Variadic(s)
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

    pub fn variadic(args: Ty, ret: Ty) -> Signature {
        VariadicSignature { args, ret }.into()
    }

    pub fn ret(&self) -> Ty {
        match self {
            Signature::Nullary(s) => s.ret,
            Signature::Unary(s) => s.ret,
            Signature::Binary(s) => s.ret,
            Signature::Ternary(s) => s.ret,
            Signature::Variadic(s) => s.ret,
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
            (Signature::Variadic(s), _) => Ok(s.args),
            _ => Err(Error::OperandUnavailable),
        }
    }

    pub fn arity(&self) -> Option<usize> {
        match self {
            Signature::Nullary(_) => Some(0),
            Signature::Unary(_) => Some(1),
            Signature::Binary(_) => Some(2),
            Signature::Ternary(_) => Some(3),
            Signature::Variadic(_) => None,
        }
    }

    pub(crate) fn flatten(&self) -> Self {
        match self {
            Signature::Nullary(s) => Signature::nullary(s.ret.flatten()),
            Signature::Unary(s) => Signature::unary(s.arg0.flatten(), s.ret.flatten()),
            Signature::Binary(s) => {
                Signature::binary(s.arg0.flatten(), s.arg1.flatten(), s.ret.flatten())
            }
            Signature::Ternary(s) => Signature::ternary(
                s.arg0.flatten(),
                s.arg1.flatten(),
                s.arg2.flatten(),
                s.ret.flatten(),
            ),
            Signature::Variadic(s) => Signature::variadic(s.args.flatten(), s.ret.flatten()),
        }
    }
}

macro_rules! operators {
    ($($t:ident,)+) => {

        paste! {
            #[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug)]
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
            #[cfg(feature = "compile")]
            pub(crate) fn sig(&self) -> &Signature {
                match self {
                    $(Operator::$t(op) => &op.sig,)+
                }
            }

            #[cfg(feature = "compile")]
            pub(crate) fn sig_mut(&mut self) -> &mut Signature {
                match self {
                    $(Operator::$t(op) => &mut op.sig,)+
                }
            }

            pub fn short_name(&self) -> &str {
                match self {
                    $(Operator::$t(op) => op.short_name(),)+
                }
            }

            pub(crate) fn get_from_textual<'a, E: 'a + nom::error::ParseError<&'a str> + nom::error::ContextError<&'a str>>(name: &'a str) -> impl FnMut(&'a str) -> std::result::Result<(&str, Operator), nom::Err<E>> {
                use crate::textual::{FromTextual, parse_operator_error};
                match name {
                    $(paste! {[<$t Op>]::SHORT_NAME} => paste! {[<$t Op>]::from_textual},)+
                    // The names below are deprecated aliases, maintained for a long period of time for compatibility
                    "PrimDeriveSeed" => DeriveSeedOp::from_textual, // pre v0.1.5
                    "PrimPrfKeyGen" => PrfKeyGenOp::from_textual, // pre v0.1.5
                    "HostMean" => MeanOp::from_textual, // pre v0.1.5
                    "FixedpointMeanOp" => MeanOp::from_textual, // pre v0.1.5
                    "FloatingpointMeanOp" => MeanOp::from_textual, // pre v0.1.5
                    "RepFixedpointMean" => MeanOp::from_textual, // pre v0.1.5
                    _ => parse_operator_error,
                }
            }
        }
    }
}

operators![
    Abs,
    Add,
    And,
    AtLeast2D,
    BitExtract,
    Broadcast,
    Cast,
    Concat,
    Constant,
    Decrypt,
    DeriveSeed,
    Div,
    Diag,
    Dot,
    ExpandDims,
    Identity,
    IndexAxis,
    Inverse,
    Input,
    Load,
    Mul,
    Mean,
    Output,
    Ones,
    Or,
    PrfKeyGen,
    Reshape,
    Receive,
    Relu,
    RingFixedpointArgmax,
    RingFixedpointDecode,
    RingFixedpointEncode,
    RingInject,
    RingFixedpointMean,
    Sample,
    SampleSeeded,
    Send,
    Save,
    Shape,
    Shl,
    Shr,
    Sign,
    Slice,
    Sqrt,
    Squeeze,
    Sub,
    Sum,
    Transpose,
    Xor,
    Zeros,
    // Fixed-point operators
    Equal,
    EqualZero,
    Exp,
    FixedpointEncode,
    FixedpointDecode,
    Greater,
    Less,
    Neg,
    Pow2,
    Sigmoid,
    // Additive operators
    AdtToRep,
    // Replicated operators
    AddN,
    Argmax,
    BitDecompose,
    BitCompose,
    Fill,
    Index,
    Log2,
    Log,
    Maximum,
    Msb,
    Mux,
    RepToAdt,
    Reveal,
    Share,
    Softmax,
    ShlDim,
    TruncPr,
    // Mirrored Operators
    Demirror,
    Mirror,
];

pub trait HasShortName {
    fn short_name(&self) -> &str;
}

// Top (logical) level ops:

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct IdentityOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual)]
pub struct SendOp {
    pub sig: Signature,
    pub rendezvous_key: RendezvousKey,
    pub receiver: Role,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual)]
pub struct ReceiveOp {
    pub sig: Signature,
    pub rendezvous_key: RendezvousKey,
    pub sender: Role,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct InputOp {
    pub sig: Signature,
    pub arg_name: String,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct OutputOp {
    pub sig: Signature,
    pub tag: String,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct LoadOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct CastOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct SaveOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName, ToTextual, FromTextual)]
pub struct ConstantOp {
    pub sig: Signature,
    pub value: Constant, // TODO Box<Constant> or Box inside Constant?
}

#[allow(clippy::derive_hash_xor_eq)]
impl Hash for ConstantOp {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.sig.hash(state);
        self.value.to_textual().hash(state)
        // TODO(Morten) we must also take `self.value` into account!
        // self.value.hash(state);
    }
}

impl std::cmp::Eq for ConstantOp {}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct AtLeast2DOp {
    pub sig: Signature,
    pub to_column_vector: bool,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct IndexAxisOp {
    pub sig: Signature,
    pub axis: usize,
    pub index: usize,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct SliceOp {
    pub sig: Signature,
    pub slice: SliceInfo,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct OnesOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct ZerosOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual)]
pub struct ExpandDimsOp {
    pub sig: Signature,
    pub axis: Vec<usize>,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct ConcatOp {
    pub sig: Signature,
    pub axis: u32,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct ReshapeOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, FromTextual)]
pub struct SqueezeOp {
    pub sig: Signature,
    pub axis: Option<usize>,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct TransposeOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct InverseOp {
    pub sig: Signature,
}

// TODO(Morten) rename to LogicalAddOp?
#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct AddOp {
    pub sig: Signature,
}

// TODO(Morten) rename to LogicalSubOp?
#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct SubOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct MulOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct DivOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct DotOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, FromTextual)]
pub struct MeanOp {
    pub sig: Signature,
    pub axis: Option<u32>,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct SigmoidOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, FromTextual)]
pub struct SumOp {
    pub sig: Signature,
    pub axis: Option<usize>,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct SignOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct SqrtOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]

pub struct ShapeOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct DiagOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual)]
pub struct BitToRingOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual)]
pub struct DeriveSeedOp {
    pub sig: Signature,
    pub sync_key: SyncKey,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct PrfKeyGenOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct DecryptOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, FromTextual)]
pub struct RingFixedpointMeanOp {
    pub sig: Signature,
    pub axis: Option<u32>,
    pub scaling_base: u64,
    pub scaling_exp: u32,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, FromTextual)]
pub struct SampleOp {
    pub sig: Signature,
    pub max_value: Option<u64>,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, FromTextual)]
pub struct SampleSeededOp {
    pub sig: Signature,
    pub max_value: Option<u64>,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct ShlOp {
    pub sig: Signature,
    pub amount: usize,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct ShrOp {
    pub sig: Signature,
    pub amount: usize,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct RingInjectOp {
    pub sig: Signature,
    pub bit_idx: usize,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct BitExtractOp {
    pub sig: Signature,
    pub bit_idx: usize,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct XorOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct AndOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct OrOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct Pow2Op {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct ExpOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct EqualOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct EqualZeroOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct LessOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct GreaterOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct FixedpointEncodeOp {
    pub sig: Signature,
    pub fractional_precision: u32,
    pub integral_precision: u32,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct FixedpointDecodeOp {
    pub sig: Signature,
    pub fractional_precision: u32,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct NegOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct RingFixedpointEncodeOp {
    pub sig: Signature,
    pub scaling_base: u64,
    pub scaling_exp: u32,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct RingFixedpointDecodeOp {
    pub sig: Signature,
    pub scaling_base: u64,
    pub scaling_exp: u32,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct RingFixedpointArgmaxOp {
    pub sig: Signature,
    // axis can be optional (in which case we need to do an argmax over axis 0)
    // TODO(Dragos) once we have shape inference we can make axis optional
    // since we can automatically get the upmost index arg
    pub axis: usize,
    pub upmost_index: usize,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct AdtToRepOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct AbsOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct ReluOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct MsbOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct ShareOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct RevealOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct AddNOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct TruncPrOp {
    pub sig: Signature,
    pub amount: u32,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct RepToAdtOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName, ToTextual, FromTextual)]
pub struct FillOp {
    pub sig: Signature,
    pub value: Constant,
}

impl std::cmp::Eq for FillOp {}

#[allow(clippy::derive_hash_xor_eq)]
impl Hash for FillOp {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.sig.hash(state);
        self.value.to_textual().hash(state);
        // TODO(Morten) we must also take `self.value` into account!
        // self.value.hash(state);
    }
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct ShlDimOp {
    pub sig: Signature,
    pub amount: usize,
    pub bit_length: usize,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct IndexOp {
    pub sig: Signature,
    pub index: usize,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct BitDecomposeOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct BitComposeOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct MuxOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct DemirrorOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct MirrorOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct MaximumOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct SoftmaxOp {
    pub sig: Signature,
    // axis can be optional (in which case we need to do a softmax over every entry)
    pub axis: usize,
    pub upmost_index: usize,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct ArgmaxOp {
    pub sig: Signature,
    // axis can be optional (in which case we need to do an argmax over axis 0)
    // TODO(Dragos) once we have shape inference we can make axis optional
    // since we can automatically get the upmost index arg
    pub axis: usize,
    pub upmost_index: usize,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct BroadcastOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct LogOp {
    pub sig: Signature,
}

#[derive(
    Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Debug, ShortName, ToTextual, FromTextual,
)]
pub struct Log2Op {
    pub sig: Signature,
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
            #[derive(Serialize, Deserialize, Hash, PartialEq, Eq, Clone, Debug)]
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
            pub(crate) fn ty(&self) -> PlacementTy {
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

placements![Host, Replicated, Additive, Mirrored3,];

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

pub trait Placed {
    type Placement;

    fn placement(&self) -> std::result::Result<Self::Placement, Error>;
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct Operation {
    pub name: String,
    pub kind: Operator,
    pub inputs: Vec<String>, // TODO(Morten) use indices instead of strings?
    pub placement: Placement,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct NamedComputation {
    pub operations: Vec<Operation>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct IndexedOperation {
    pub operator: usize,
    pub inputs: Vec<usize>,
    pub placement: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct IndexedComputation {
    pub operations: Vec<IndexedOperation>,
    pub operators: Vec<Operator>,
    pub placements: Vec<Placement>,
}

impl TryFrom<&NamedComputation> for IndexedComputation {
    type Error = Error;

    fn try_from(computation: &NamedComputation) -> Result<IndexedComputation> {
        let unique_placements = computation
            .operations
            .iter()
            .map(|op| &op.placement)
            .collect::<HashSet<&Placement>>();
        let placements: Vec<Placement> = unique_placements.into_iter().cloned().collect();
        let placements_map: HashMap<&Placement, usize> = placements
            .iter()
            .enumerate()
            .map(|(i, plc)| (plc, i))
            .collect();

        let unique_operators = computation
            .operations
            .iter()
            .map(|op| &op.kind)
            .collect::<HashSet<&Operator>>();
        let operators: Vec<Operator> = unique_operators.into_iter().cloned().collect();
        let operators_map: HashMap<&Operator, usize> = operators
            .iter()
            .enumerate()
            .map(|(i, op)| (op, i))
            .collect();

        let op_names_map: HashMap<&String, usize> = computation
            .operations
            .iter()
            .enumerate()
            .map(|(i, op)| (&op.name, i))
            .collect();

        let operations = computation
            .operations
            .iter()
            .map(|op| {
                let inputs = op
                    .inputs
                    .iter()
                    .map(|name| {
                        op_names_map.get(name).cloned().ok_or_else(|| {
                            Error::MalformedComputation(format!(
                                "Missing operation '{}' used as an operand for '{}'",
                                name, op.name
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;

                Ok(IndexedOperation {
                    inputs,
                    operator: operators_map[&op.kind], // should be there by construction
                    placement: placements_map[&op.placement], // should be there by construction
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(IndexedComputation {
            operators,
            operations,
            placements,
        })
    }
}

impl TryFrom<&IndexedComputation> for Computation {
    type Error = Error;

    fn try_from(compact: &IndexedComputation) -> Result<NamedComputation> {
        let operations = compact
            .operations
            .iter()
            .enumerate()
            .map(|(i, op)| {
                let kind = compact.operators.get(op.operator).cloned().ok_or_else(|| {
                    Error::MalformedComputation(format!(
                        "Missing operator with index {}",
                        op.operator
                    ))
                })?;

                let inputs = op
                    .inputs
                    .iter()
                    .map(|inp| format!("op_{:?}", inp))
                    .collect();

                let placement = compact
                    .placements
                    .get(op.placement)
                    .cloned()
                    .ok_or_else(|| {
                        Error::MalformedComputation(format!(
                            "Missing placement with index {}",
                            op.placement
                        ))
                    })?;

                Ok(Operation {
                    name: format!("op_{:?}", i),
                    kind,
                    inputs,
                    placement,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(NamedComputation { operations })
    }
}

#[derive(Debug)]
pub struct OperationIndex {
    pub index: usize,
}

impl NamedComputation {
    #[tracing::instrument(skip(bytes))]
    pub fn from_msgpack(bytes: &[u8]) -> Result<Self> {
        rmp_serde::from_slice(bytes).map_err(|e| Error::SerializationError(e.to_string()))
    }

    #[tracing::instrument(skip(self))]
    pub fn to_msgpack(&self) -> Result<Vec<u8>> {
        rmp_serde::to_vec(self).map_err(|e| Error::SerializationError(e.to_string()))
    }

    #[tracing::instrument(skip(comp))]
    pub fn from_textual(comp: &str) -> Result<Self> {
        crate::textual::parallel_parse_computation(comp, 12)
            .map_err(|e| Error::SerializationError(e.to_string()))
    }

    #[tracing::instrument(skip(self, path))]
    pub fn write_textual<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| Error::SerializationError(e.to_string()))?;

        let mut file = BufWriter::new(file);

        for op in self.operations.iter() {
            let op_textual = op.to_textual();
            writeln!(file, "{}", op_textual)
                .map_err(|e| Error::SerializationError(e.to_string()))?;
        }

        file.flush()
            .map_err(|e| Error::SerializationError(e.to_string()))?;

        Ok(())
    }

    #[tracing::instrument(skip(bytes))]
    pub fn from_bincode<B: AsRef<[u8]>>(bytes: B) -> Result<Self> {
        bincode::deserialize(bytes.as_ref()).map_err(|e| Error::SerializationError(e.to_string()))
    }

    #[tracing::instrument(skip(self))]
    pub fn to_bincode(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| Error::SerializationError(e.to_string()))
    }

    #[deprecated]
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::from_msgpack(&bytes)
    }

    #[deprecated]
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.to_msgpack()
    }

    pub fn from_disk<P: AsRef<Path>>(path: P) -> Result<Self> {
        let p = path.as_ref();
        let f = File::open(p).map_err(|e| {
            Error::Unexpected(Some(format!(
                "File not found error for path {0}. Original: {1}.",
                p.display(),
                e
            )))
        })?;
        rmp_serde::decode::from_read(f).map_err(|e| Error::SerializationError(e.to_string()))
    }

    pub fn to_disk<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut file_buffer =
            File::create(path).map_err(|e| Error::SerializationError(e.to_string()))?;
        rmp_serde::encode::write(&mut file_buffer, self)
            .map_err(|e| Error::SerializationError(e.to_string()))?;
        Ok(())
    }

    /// Compute the dataflow graph associated with a computation.
    ///
    /// Edges indicate the direction in which data flows, i.e. from producer to consumer.
    pub fn as_graph(&self) -> Graph<OperationIndex, ()> {
        let exact_node_count = self.operations.len();
        let rough_edge_count = self.operations.len() * 2; // assume roughly two inputs on average

        let mut graph = Graph::with_capacity(exact_node_count, rough_edge_count);

        let mut vertex_map: HashMap<&str, NodeIndex> = HashMap::with_capacity(exact_node_count);

        let mut send_nodes: HashMap<&RendezvousKey, NodeIndex> = HashMap::new();
        let mut recv_nodes: HashMap<&RendezvousKey, NodeIndex> = HashMap::new();

        let mut rdv_keys: HashSet<&RendezvousKey> = HashSet::new();

        for (index, op) in self.operations.iter().enumerate() {
            let vertex = graph.add_node(OperationIndex { index });
            match op.kind {
                Operator::Send(ref op) => {
                    let key = &op.rendezvous_key;

                    if send_nodes.contains_key(key) {
                        Error::MalformedComputation(format!(
                            "Already had a send node with same rdv key at key {}",
                            key
                        ));
                    }

                    send_nodes.insert(key, vertex);
                    rdv_keys.insert(key);
                }
                Operator::Receive(ref op) => {
                    let key = &op.rendezvous_key;

                    if recv_nodes.contains_key(key) {
                        Error::MalformedComputation(format!(
                            "Already had a recv node with same rdv key at key {}",
                            key
                        ));
                    }

                    recv_nodes.insert(key, vertex);
                    rdv_keys.insert(key);
                }
                _ => {}
            }
            vertex_map.insert(&op.name, vertex);
        }

        for op in self.operations.iter() {
            for ins in op.inputs.iter() {
                graph.add_edge(vertex_map[&ins.as_ref()], vertex_map[&op.name.as_ref()], ());
            }
        }

        for key in rdv_keys.into_iter() {
            if !send_nodes.contains_key(key) {
                Error::MalformedComputation(format!("No send node with rdv key {}", key));
            }
            if !recv_nodes.contains_key(key) {
                Error::MalformedComputation(format!("No recv node with rdv key {}", key));
            }
            // add edge send->recv (send must be evaluated before recv)
            graph.add_edge(send_nodes[key], recv_nodes[key], ());
        }

        graph
    }
}

pub type Computation = NamedComputation;

mod tests {
    #![allow(unused_imports)]
    use super::*;

    #[test]
    fn test_session_id() {
        let session_id_str = "01FGSQ37YDJSVJXSA6SSY7G4Y2";
        let session_id = SessionId::try_from(session_id_str).unwrap();
        let expected: [u8; 16] = [
            240, 33, 112, 148, 149, 223, 59, 54, 60, 236, 245, 54, 147, 48, 64, 108,
        ];
        assert_eq!(session_id.logical, session_id_str);
        assert_eq!(session_id.to_string(), session_id_str);
        assert_eq!(session_id.secure, expected);
        assert_eq!(*session_id.as_bytes(), expected);

        let session_id_str = "hello world";
        let session_id = SessionId::try_from(session_id_str).unwrap();
        let expected: [u8; 16] = [
            215, 73, 129, 239, 167, 10, 12, 136, 11, 141, 140, 25, 133, 208, 117, 219,
        ];
        assert_eq!(session_id.logical, session_id_str);
        assert_eq!(session_id.to_string(), session_id_str);
        assert_eq!(session_id.secure, expected);
        assert_eq!(*session_id.as_bytes(), expected);
    }

    #[test]
    fn test_binary_roundtrip() {
        use std::convert::TryInto;
        let original: Computation = r#"constant_0 = Constant{value = HostFloat64Tensor([[0.12131529]])}: () -> Tensor<Float64> () @Host(player2)
        cast_0 = Cast: (Tensor<Float64>) -> Tensor<Fixed128(24, 40)> (constant_0) @Host(player2)
        x = Input{arg_name = "x"}: () -> AesTensor () @Host(player0)
        key = Input{arg_name = "key"}: () -> AesKey () @Replicated(player0, player1, player2)
        decrypt_0 = Decrypt: (AesKey, AesTensor) -> Tensor<Fixed128(24, 40)> (key, x) @Replicated(player0, player1, player2)
        dot_0 = Dot: (Tensor<Fixed128(24, 40)>, Tensor<Fixed128(24, 40)>) -> Tensor<Fixed128(24, 40)> (decrypt_0, cast_0) @Replicated(player0, player1, player2)
        cast_1 = Cast: (Tensor<Fixed128(24, 40)>) -> Tensor<Float64> (dot_0) @Host(player1)
        output_0 = Output{tag = "output_0"}: (Tensor<Float64>) -> Tensor<Float64> (cast_1) @Host(player1)"#.try_into().unwrap();
        let bytes = original.to_msgpack().unwrap();
        let read_back = Computation::from_msgpack(&bytes).unwrap();
        assert_eq!(original.operations, read_back.operations);
    }

    #[test]
    fn test_write_textual() {
        use std::convert::TryInto;
        use std::fs::read_to_string;
        use tempfile::tempdir;
        let original: Computation = r#"constant_0 = Constant{value = HostFloat64Tensor([[0.12131529]])}: () -> Tensor<Float64> () @Host(player2)
        cast_0 = Cast: (Tensor<Float64>) -> Tensor<Fixed128(24, 40)> (constant_0) @Host(player2)
        x = Input{arg_name = "x"}: () -> AesTensor () @Host(player0)
        key = Input{arg_name = "key"}: () -> AesKey () @Replicated(player0, player1, player2)
        decrypt_0 = Decrypt: (AesKey, AesTensor) -> Tensor<Fixed128(24, 40)> (key, x) @Replicated(player0, player1, player2)
        dot_0 = Dot: (Tensor<Fixed128(24, 40)>, Tensor<Fixed128(24, 40)>) -> Tensor<Fixed128(24, 40)> (decrypt_0, cast_0) @Replicated(player0, player1, player2)
        cast_1 = Cast: (Tensor<Fixed128(24, 40)>) -> Tensor<Float64> (dot_0) @Host(player1)
        output_0 = Output{tag = "output_0"}: (Tensor<Float64>) -> Tensor<Float64> (cast_1) @Host(player1)"#.try_into().unwrap();
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("temp_comp.moose");
        original.write_textual(file_path.clone()).unwrap();
        let source = read_to_string(file_path).unwrap();
        let read_back = Computation::from_textual(&source).unwrap();
        assert_eq!(original.operations, read_back.operations);
    }
}
