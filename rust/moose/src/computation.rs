use crate::additive::{Additive128Tensor, Additive64Tensor};
use crate::bit::BitTensor;
use crate::error::{Error, Result};
use crate::fixedpoint::{Fixed128Tensor, Fixed64Tensor};
use crate::kernels::{ConcreteContext, Context};
use crate::prim::{Nonce, PrfKey, RawNonce, RawPrfKey, RawSeed, Seed};
use crate::replicated::{
    Replicated128Tensor, Replicated64Tensor, ReplicatedBitTensor, ReplicatedSetup,
};
use crate::ring::{Ring128Tensor, Ring64Tensor};
use crate::standard::{
    Float32Tensor, Float64Tensor, Int16Tensor, Int32Tensor, Int64Tensor, Int8Tensor, RawShape,
    Shape, Uint16Tensor, Uint32Tensor, Uint64Tensor, Uint8Tensor,
};
use derive_more::Display;
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

pub trait KnownType<C: Context> {
    type Type;
    const TY: Ty;
}

// Primitives are trivial values. They are what can live on the nodes of the computation graph.
// Primitive can not be a Unit, an Unknown or a complex structure such as ReplicatedTensor.
macro_rules! primitives {
    ($($val:ident $($t:ident)?,)+) => {

        #[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
        pub enum Primitive {
            $($val($val),)+
            // TODO promote below to match other values
            Float32(f32),
            Float64(f64),
            Ring64(u64),
            Ring128(u128),
        }

        impl Primitive {
            pub fn ty(&self) -> Ty {
                match self {
                    $(Primitive::$val(_) => primitives!(@ty $val $($t)?),)+
                    // TODO promote below to match other values
                    Primitive::Float32(_) => Ty::Float32,
                    Primitive::Float64(_) => Ty::Float64,
                    Primitive::Ring64(_) => Ty::Ring64,
                    Primitive::Ring128(_) => Ty::Ring128,
                }
            }

            pub fn place(&self, plc: &HostPlacement) -> Value {
                match self {
                    $(
                        Primitive::$val(x) => {primitives!(@value(x.clone(), plc.clone().into()) $val $($t)?)},
                    )+
                    // TODO promote below to match other values
                    Primitive::Float32(x) => Value::Float32(x.clone()),
                    Primitive::Float64(x) => Value::Float64(x.clone()),
                    Primitive::Ring64(x) => Value::Ring64(x.clone()),
                    Primitive::Ring128(x) => Value::Ring128(x.clone()),
                }
            }
        }

        $(
        impl From<$val> for Primitive {
            fn from(x: $val) -> Self {
                Primitive::$val(x)
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
// The lines with 1 identifier are for linking to the "Unplaced" values, where the Primitive and Value are essentially the same and can be converted easily.
primitives![
    RawShape Shape,
    RawSeed Seed,
    RawPrfKey PrfKey,
    RawNonce Nonce,
    String,
    BitTensor,
    Ring64Tensor,
    Ring128Tensor,
    Float32Tensor,
    Float64Tensor,
    Int8Tensor,
    Int16Tensor,
    Int32Tensor,
    Int64Tensor,
    Uint8Tensor,
    Uint16Tensor,
    Uint32Tensor,
    Uint64Tensor,
];

// Values are anything that can flow along the edges of the computation graph.
// Some values are just placed primitives, but some could be more complex.
macro_rules! values {
    ($($val:ident,)+) => {

        #[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
        pub enum Value {
            $($val($val),)+
            // TODO promote below to match other values
            Unit,
            Float32(f32),
            Float64(f64),
            Ring64(u64),
            Ring128(u128),
        }

        #[derive(Serialize, Deserialize, PartialEq, Eq, Copy, Clone, Debug, Display)]
        pub enum Ty {
            Unknown,
            $($val,)+
            // TODO promote below to match other values
            Unit,
            Float32,
            Float64,
            Ring64,
            Ring128,
        }

        impl Value {
            pub fn ty(&self) -> Ty {
                match self {
                    $(Value::$val(_) => Ty::$val,)+
                    // TODO promote below to match other values
                    Value::Unit => Ty::Unit,
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
        impl KnownType<ConcreteContext> for $val {
            type Type = $val;
            const TY: Ty = Ty::$val;
        }
        )+
    };
}

values![
    Shape,
    Seed,
    PrfKey,
    Nonce,
    String,
    BitTensor,
    Ring64Tensor,
    Ring128Tensor,
    Float32Tensor,
    Float64Tensor,
    Int8Tensor,
    Int16Tensor,
    Int32Tensor,
    Int64Tensor,
    Uint8Tensor,
    Uint16Tensor,
    Uint32Tensor,
    Uint64Tensor,
    Fixed64Tensor,
    Fixed128Tensor,
    Replicated64Tensor,
    Replicated128Tensor,
    ReplicatedBitTensor,
    ReplicatedSetup,
    Additive64Tensor,
    Additive128Tensor,
];

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
    StdAdd,
    StdSub,
    StdMul,
    StdDiv,
    StdDot,
    StdMean,
    StdExpandDims,
    StdReshape,
    StdAtLeast2D,
    StdShape,
    StdSlice,
    StdSum,
    StdOnes,
    StdConcatenate,
    StdTranspose,
    StdInverse,
    RingAdd,
    RingSub,
    RingMul,
    RingDot,
    RingSum,
    RingShape,
    RingSample,
    RingFill,
    RingShl,
    RingShr,
    RingInject,
    BitExtract,
    BitSample,
    BitFill,
    BitXor,
    BitAnd,
    PrimDeriveSeed,
    PrimPrfKeyGen,
    FixedAdd,
    FixedMul,
    FixedpointRingEncode,
    FixedpointRingDecode,
    FixedpointRingMean,
    AdtReveal,
    AdtAdd,
    AdtMul,
    RepSetup,
    RepShare,
    RepReveal,
    RepAdd,
    RepMul,
    RepToAdt,
];

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct SendOp {
    pub sig: Signature,
    pub rendezvous_key: String,
    pub receiver: Role,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct IdentityOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct ReceiveOp {
    pub sig: Signature,
    pub rendezvous_key: String,
    pub sender: Role,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct InputOp {
    pub sig: Signature,
    pub arg_name: String,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct OutputOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct LoadOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct SaveOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct ConstantOp {
    pub sig: Signature,
    pub value: Primitive, // TODO Box<Primitive> or Box inside Primitive?
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct StdAddOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct StdSubOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct StdMulOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct StdDivOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct StdDotOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct StdMeanOp {
    pub sig: Signature,
    pub axis: Option<u32>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct StdOnesOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct StdConcatenateOp {
    pub sig: Signature,
    pub axis: u32,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct StdAtLeast2DOp {
    pub sig: Signature,
    pub to_column_vector: bool,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct StdExpandDimsOp {
    pub sig: Signature,
    pub axis: u32,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct StdReshapeOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct StdShapeOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct StdSliceOp {
    pub sig: Signature,
    pub start: u32,
    pub end: u32,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct StdSumOp {
    pub sig: Signature,
    pub axis: Option<u32>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct StdTransposeOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct StdInverseOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct PrimDeriveSeedOp {
    pub sig: Signature,
    pub nonce: RawNonce,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct PrimPrfKeyGenOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RingAddOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RingSubOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RingMulOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RingDotOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RingSumOp {
    pub sig: Signature,
    pub axis: Option<u32>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RingShapeOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RingFillOp {
    pub sig: Signature,
    pub value: Primitive,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RingSampleOp {
    pub sig: Signature,
    pub max_value: Option<u64>,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RingShlOp {
    pub sig: Signature,
    pub amount: usize,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RingShrOp {
    pub sig: Signature,
    pub amount: usize,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RingInjectOp {
    pub sig: Signature,
    pub bit_idx: usize,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct BitExtractOp {
    pub sig: Signature,
    pub bit_idx: usize,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct BitSampleOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct BitFillOp {
    pub sig: Signature,
    pub value: u8,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct BitXorOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct BitAndOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct FixedAddOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct FixedMulOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct FixedpointRingEncodeOp {
    pub sig: Signature,
    pub scaling_base: u64,
    pub scaling_exp: u32,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct FixedpointRingDecodeOp {
    pub sig: Signature,
    pub scaling_base: u64,
    pub scaling_exp: u32,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct FixedpointRingMeanOp {
    pub sig: Signature,
    pub axis: Option<usize>,
    pub scaling_base: u64,
    pub scaling_exp: u32,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct AdtRevealOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct AdtAddOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct AdtMulOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RepSetupOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RepShareOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RepRevealOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RepAddOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RepMulOp {
    pub sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RepToAdtOp {
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

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
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

    fn placement(&self) -> Self::Placement;
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
