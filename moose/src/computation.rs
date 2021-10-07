use crate::additive::{
    AdditiveBitTensor, AdditiveRing128Tensor, AdditiveRing64Tensor, AdditiveShape,
};
use crate::error::{Error, Result};
use crate::fixedpoint::{Fixed128Tensor, Fixed64Tensor};
use crate::floatingpoint::{Float32Tensor, Float64Tensor};
use crate::host::{
    HostBitArray128, HostBitArray64, HostBitTensor, HostFixed128Tensor, HostFixed64Tensor,
    HostFloat32Tensor, HostFloat64Tensor, HostInt16Tensor, HostInt32Tensor, HostInt64Tensor,
    HostInt8Tensor, HostRing128Tensor, HostRing64Tensor, HostShape, HostString, HostUint16Tensor,
    HostUint32Tensor, HostUint64Tensor, HostUint8Tensor, RawShape, SliceInfo,
};
use crate::kernels::Session;
use crate::logical::{Tensor, TensorDType};
use crate::prim::{PrfKey, RawPrfKey, RawSeed, Seed, SyncKey};
use crate::replicated::{
    ReplicatedBitArray128, ReplicatedBitArray64, ReplicatedBitTensor, ReplicatedFixed128Tensor,
    ReplicatedFixed64Tensor, ReplicatedRing128Tensor, ReplicatedRing64Tensor, ReplicatedSetup,
    ReplicatedShape,
};
use crate::symbolic::Symbolic;
use byteorder::{ByteOrder, LittleEndian};
use derive_more::Display;
use macros::ShortName;
use paste::paste;
use serde::{Deserialize, Serialize};
use sodiumoxide::crypto::generichash;
use std::convert::TryFrom;

pub const TAG_BYTES: usize = 128 / 8;
static_assertions::const_assert!(TAG_BYTES >= sodiumoxide::crypto::generichash::DIGEST_MIN);
static_assertions::const_assert!(TAG_BYTES <= sodiumoxide::crypto::generichash::DIGEST_MAX);

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
        let mut raw = [0; TAG_BYTES];
        sodiumoxide::init().expect("failed to initialize sodiumoxide");
        sodiumoxide::randombytes::randombytes_into(&mut raw);
        RendezvousKey(raw)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
        sodiumoxide::init().map_err(|e| {
            crate::error::Error::Unexpected(Some(format!(
                "failed to initialize sodiumoxide: {:?}",
                e
            )))
        })?;
        let digest = generichash::hash(s.as_bytes(), Some(TAG_BYTES), None).map_err(|e| {
            crate::error::Error::Unexpected(Some(format!(
                "failed to hash session ID: {}: {:?}",
                s, e
            )))
        })?;
        let mut raw_hash = [0u8; TAG_BYTES];
        raw_hash.copy_from_slice(digest.as_ref());
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
        let mut raw = [0; TAG_BYTES];
        sodiumoxide::init().expect("failed to initialize sodiumoxide");
        sodiumoxide::randombytes::randombytes_into(&mut raw);
        let hex_vec: Vec<String> = raw.iter().map(|byte| format!("{:02X}", byte)).collect();
        let hex_string = hex_vec.join("");
        SessionId {
            logical: hex_string,
            secure: raw,
        }
    }
}

pub trait SymbolicType {
    type Type;
}

pub trait CanonicalType {
    type Type;
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
    RawSeed Seed,
    RawPrfKey PrfKey,
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

// Values are anything that can flow along the edges of the computation graph.
// Some values are just placed constants, but some could be more complex.
macro_rules! values {
    ($($val:ident$(($inner:ident::$default:ident))?,)+) => {

        #[derive(Serialize, Deserialize, PartialEq, Eq, Copy, Clone, Debug, Display)]
        pub enum Ty {
            Unknown,
            $($val$(($inner))?,)+
            // TODO promote below to match other values
            Bit,
            Float32,
            Float64,
            Ring64,
            Ring128,
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
        impl KnownType<crate::kernels::SyncSession> for $val {
            type Type = $val;
            const TY: Ty = Ty::$val$(($inner::$default))?;
        }
        )+

        #[derive(PartialEq, Clone, Debug)]
        #[allow(clippy::large_enum_variant)]
        pub enum SymbolicValue {
            $($val(Box<<$val as SymbolicType>::Type>),)+
        }

        impl SymbolicValue {
            pub fn ty(&self) -> Ty {
                match self {
                    $(SymbolicValue::$val(_) => Ty::$val$(($inner::$default))?,)+
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
        impl From<<$val as SymbolicType>::Type> for SymbolicValue {
            fn from(x: <$val as SymbolicType>::Type) -> Self {
                SymbolicValue::$val(Box::new(x))
            }
        }
        )+

        $(
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
        impl KnownType<crate::symbolic::SymbolicSession> for $val {
            type Type = <$val as SymbolicType>::Type;
            const TY: Ty = Ty::$val$(($inner::$default))?;
        }
        )+
    };
}

values![
    Unit,
    HostShape,
    Seed,
    PrfKey,
    HostString,
    Tensor(TensorDType::Unknown),
    HostBitTensor,
    HostBitArray64,
    HostBitArray128,
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
    Fixed64Tensor,
    Fixed128Tensor,
    Float32Tensor,
    Float64Tensor,
    ReplicatedRing64Tensor,
    ReplicatedRing128Tensor,
    ReplicatedBitTensor,
    ReplicatedBitArray64,
    ReplicatedBitArray128,
    ReplicatedFixed64Tensor,
    ReplicatedFixed128Tensor,
    ReplicatedSetup,
    ReplicatedShape,
    AdditiveBitTensor,
    AdditiveRing64Tensor,
    AdditiveRing128Tensor,
    AdditiveShape,
];

// A macros to define something common for all the possible values
#[macro_export]
macro_rules! for_all_values {( $($rules:tt)* ) => (
    macro_rules! __emit__ { $($rules)* }
    __emit__! {
        HostString,
        Unit,
        HostShape,
        Seed,
        PrfKey,
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
        HostFixed64Tensor,
        HostFixed128Tensor
    }
)}

// Unit is still special. Placed unit is just a host placement.
#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct Unit(pub HostPlacement);

impl SymbolicType for Unit {
    type Type = Symbolic<Unit>;
}

impl Placed for Unit {
    type Placement = HostPlacement;

    fn placement(&self) -> Result<Self::Placement> {
        Ok(self.0.clone())
    }
}

impl Ty {
    pub fn flatten(&self) -> Ty {
        match self {
            Ty::Tensor(_) => Ty::Tensor(TensorDType::Unknown),
            _ => *self,
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Copy, Clone, Debug)]
pub enum Signature {
    Nullary(NullarySignature),
    Unary(UnarySignature),
    Binary(BinarySignature),
    Ternary(TernarySignature),
    Variadic(VariadicSignature),
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

#[derive(Serialize, Deserialize, PartialEq, Copy, Clone, Debug)]
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

    pub fn flatten(&self) -> Self {
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

    pub fn merge(&mut self, another: Signature) -> anyhow::Result<()> {
        match (self, &another) {
            (Signature::Nullary(s), Signature::Nullary(o)) => s.merge(o),
            (Signature::Unary(s), Signature::Unary(o)) => s.merge(o),
            (Signature::Binary(s), Signature::Binary(o)) => s.merge(o),
            (Signature::Ternary(s), Signature::Ternary(o)) => s.merge(o),
            (Signature::Variadic(s), o) => s.merge(o),

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

impl VariadicSignature {
    pub fn merge(&mut self, another: &Signature) -> anyhow::Result<()> {
        match another {
            Signature::Variadic(sig) => {
                if let Some(new_type) = self.args.merge(&sig.args) {
                    self.args = new_type;
                }
                if let Some(new_type) = self.ret.merge(&sig.ret) {
                    self.ret = new_type;
                }
                Ok(())
            }
            Signature::Unary(sig) => {
                if self.args == sig.arg0 {
                    if let Some(new_type) = self.args.merge(&sig.arg0) {
                        self.args = new_type;
                    }
                }

                if let Some(new_type) = self.ret.merge(&sig.ret) {
                    self.ret = new_type;
                }
                Ok(())
            }
            Signature::Binary(sig) => {
                if self.args == sig.arg0 && self.args == sig.arg1 {
                    if let Some(new_type) = self.args.merge(&sig.arg0) {
                        self.args = new_type;
                    }

                    if let Some(new_type) = self.args.merge(&sig.arg1) {
                        self.args = new_type;
                    }
                }

                if let Some(new_type) = self.ret.merge(&sig.ret) {
                    self.ret = new_type;
                }

                Ok(())
            }
            o => Err(anyhow::anyhow!(
                "Can not merge {:?} with an incompatible signature {:?}",
                self,
                o
            )),
        }
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
    Cast,
    Load,
    Save,
    Send,
    Receive,
    Input,
    Output,
    Constant,
    Shape,
    PrimDeriveSeed,
    PrimPrfKeyGen,
    AtLeast2D,
    Slice,
    Ones,
    ExpandDims,
    Concat,
    Transpose,
    Dot,
    Inverse,
    Add,
    Sub,
    Mul,
    Mean,
    Sum,
    Div,
    // Host operators
    HostAdd,
    HostSub,
    HostMul,
    HostDiv,
    HostDot,
    HostMean,
    HostSqrt,
    HostExpandDims,
    HostSlice,
    HostDiag,
    HostIndexAxis,
    HostBitDec,
    HostReshape,
    HostSqueeze,
    HostSum,
    HostOnes,
    HostConcat,
    HostTranspose,
    HostInverse,
    HostAtLeast2D,
    HostShlDim,
    RingAdd,
    RingSub,
    RingNeg,
    RingMul,
    RingDot,
    RingSum,
    RingFixedpointMean,
    RingFixedpointEncode,
    RingFixedpointDecode,
    RingSample,
    RingSampleSeeded,
    RingShl,
    RingShr,
    RingInject,
    RingFill,
    BitFill,
    BitExtract,
    BitSample,
    BitSampleSeeded,
    BitXor,
    BitAnd,
    // Fixed-point operators
    FixedpointEncode,
    FixedpointDecode,
    FixedpointAdd,
    FixedpointSub,
    FixedpointMul,
    FixedpointDot,
    FixedpointTruncPr,
    FixedpointMean,
    FixedpointSum,
    // Floating-point operators
    FloatingpointAdd,
    FloatingpointSub,
    FloatingpointMul,
    FloatingpointDiv,
    FloatingpointDot,
    FloatingpointAtLeast2D,
    FloatingpointOnes,
    FloatingpointConcat,
    FloatingpointExpandDims,
    FloatingpointTranspose,
    FloatingpointInverse,
    FloatingpointMean,
    FloatingpointSum,
    // Additive operators
    AdtReveal,
    AdtFill,
    AdtAdd,
    AdtSub,
    AdtMul,
    AdtShl,
    AdtToRep,
    // Replicated operators
    RepAbs,
    RepSetup,
    RepShare,
    RepReveal,
    RepFill,
    RepAdd,
    RepSub,
    RepMul,
    RepMsb,
    RepDot,
    RepFixedpointMean,
    RepShl,
    RepSum,
    RepTruncPr,
    RepToAdt,
    RepIndexAxis,
    RepIndex,
    RepDiag,
    RepSlice,
    RepBitDec,
    RepShlDim,
    RepEqual,
];

pub trait HasShortName {
    fn short_name(&self) -> &str;
}

// Top (logical) level ops:

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct IdentityOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct SendOp {
    pub sig: Signature,
    pub rendezvous_key: RendezvousKey,
    pub receiver: Role,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct ReceiveOp {
    pub sig: Signature,
    pub rendezvous_key: RendezvousKey,
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
pub struct CastOp {
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
pub struct AtLeast2DOp {
    pub sig: Signature,
    pub to_column_vector: bool,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct SliceOp {
    pub sig: Signature,
    pub slice: SliceInfo,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct OnesOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct ExpandDimsOp {
    pub sig: Signature,
    pub axis: Vec<u32>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct ConcatOp {
    pub sig: Signature,
    pub axis: u32,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct TransposeOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct InverseOp {
    pub sig: Signature,
}

// TODO(Morten) rename to LogicalAddOp?
#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct AddOp {
    pub sig: Signature,
}

// TODO(Morten) rename to LogicalSubOp?
#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct SubOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct MulOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct DivOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct DotOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct MeanOp {
    pub sig: Signature,
    pub axis: Option<u32>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct SumOp {
    pub sig: Signature,
    pub axis: Option<u32>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct HostAddOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct HostSubOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct HostMulOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct HostDivOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct HostDotOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct HostMeanOp {
    pub sig: Signature,
    pub axis: Option<u32>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct HostSqrtOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct HostOnesOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct HostConcatOp {
    pub sig: Signature,
    pub axis: u32,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct HostAtLeast2DOp {
    pub sig: Signature,
    pub to_column_vector: bool,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct HostExpandDimsOp {
    pub sig: Signature,
    pub axis: Vec<u32>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct HostSqueezeOp {
    pub sig: Signature,
    pub axis: Option<u32>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct HostReshapeOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct HostSumOp {
    pub sig: Signature,
    pub axis: Option<u32>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct HostTransposeOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct HostInverseOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct ShapeOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct HostSliceOp {
    pub sig: Signature,
    pub slice: SliceInfo,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct HostDiagOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct HostIndexAxisOp {
    pub sig: Signature,
    pub axis: usize,
    pub index: usize,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct HostShlDimOp {
    pub sig: Signature,
    pub amount: usize,
    pub bit_length: usize,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct HostBitDecOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct BitFillOp {
    pub sig: Signature,
    pub value: Constant,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct BitToRingOp {
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
    pub sync_key: SyncKey,
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
pub struct RingFixedpointMeanOp {
    pub sig: Signature,
    pub axis: Option<u32>,
    pub scaling_base: u64,
    pub scaling_exp: u32,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RingSampleOp {
    pub sig: Signature,
    pub max_value: Option<u64>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RingSampleSeededOp {
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
pub struct BitSampleSeededOp {
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
pub struct BitNegOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FixedpointEncodeOp {
    pub sig: Signature,
    pub fractional_precision: u32,
    pub integral_precision: u32,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FixedpointDecodeOp {
    pub sig: Signature,
    pub fractional_precision: u32,
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
    pub precision: u32, // TODO(Morten) rename to amount?
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FixedpointMeanOp {
    pub sig: Signature,
    pub axis: Option<u32>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FixedpointSumOp {
    pub sig: Signature,
    pub axis: Option<u32>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FloatingpointAddOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FloatingpointSubOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FloatingpointMulOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FloatingpointDivOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FloatingpointDotOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FloatingpointAtLeast2DOp {
    pub sig: Signature,
    pub to_column_vector: bool,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FloatingpointOnesOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FloatingpointConcatOp {
    pub sig: Signature,
    pub axis: u32,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FloatingpointExpandDimsOp {
    pub sig: Signature,
    pub axis: Vec<u32>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FloatingpointTransposeOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FloatingpointInverseOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FloatingpointMeanOp {
    pub sig: Signature,
    pub axis: Option<u32>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct FloatingpointSumOp {
    pub sig: Signature,
    pub axis: Option<u32>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RingFixedpointEncodeOp {
    pub sig: Signature,
    pub scaling_base: u64,
    pub scaling_exp: u32,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RingFixedpointDecodeOp {
    pub sig: Signature,
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
pub struct RepFixedpointMeanOp {
    pub sig: Signature,
    pub axis: Option<u32>,
    pub scaling_base: u64,
    pub scaling_exp: u32,
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

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RepShlOp {
    pub sig: Signature,
    pub amount: usize,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RepShlDimOp {
    pub sig: Signature,
    pub amount: usize,
    pub bit_length: usize,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RepIndexAxisOp {
    pub sig: Signature,
    pub axis: usize,
    pub index: usize,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RepIndexOp {
    pub sig: Signature,
    pub index: usize,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RepDiagOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RepSliceOp {
    pub sig: Signature,
    pub slice: SliceInfo,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RepBitDecOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
pub struct RepEqualOp {
    pub sig: Signature,
}

pub struct RepIfElseOp {
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

mod tests {
    #![allow(unused_imports)]
    use super::*;

    #[test]
    fn test_session_id() {
        let session_id_str = "01FGSQ37YDJSVJXSA6SSY7G4Y2";
        let session_id = SessionId::try_from(session_id_str).unwrap();
        let expected: [u8; 16] = [
            155, 66, 92, 119, 188, 62, 148, 202, 13, 176, 137, 43, 64, 190, 251, 182,
        ];
        assert_eq!(session_id.logical, session_id_str);
        assert_eq!(session_id.to_string(), session_id_str);
        assert_eq!(session_id.secure, expected);
        assert_eq!(*session_id.as_bytes(), expected);

        let session_id_str = "hello world";
        let session_id = SessionId::try_from(session_id_str).unwrap();
        let expected: [u8; 16] = [
            233, 168, 4, 178, 229, 39, 253, 54, 1, 210, 255, 192, 187, 2, 60, 214,
        ];
        assert_eq!(session_id.logical, session_id_str);
        assert_eq!(session_id.to_string(), session_id_str);
        assert_eq!(session_id.secure, expected);
        assert_eq!(*session_id.as_bytes(), expected);
    }
}
