use crate::additive::{Additive128Tensor, Additive64Tensor};
use crate::bit::BitTensor;
use crate::error::{Error, Result};
use crate::fixedpoint::{Fixed128Tensor, Fixed64Tensor};
use crate::prim::{Nonce, PrfKey, Seed};
use crate::replicated::{
    Replicated128Tensor, Replicated64Tensor, ReplicatedBitTensor, ReplicatedSetup,
};
use crate::ring::{Ring128Tensor, Ring64Tensor};
use crate::standard::{
    Float32Tensor, Float64Tensor, Int16Tensor, Int32Tensor, Int64Tensor, Int8Tensor, Shape,
    Uint16Tensor, Uint32Tensor, Uint64Tensor, Uint8Tensor,
};
use derive_more::Display;
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

#[derive(Serialize, Deserialize, PartialEq, Copy, Clone, Debug, Display)]
pub enum Ty {
    Unknown,
    Unit,
    String,
    Shape,
    Seed,
    PrfKey,
    Nonce,
    Float32,
    Float64,
    Ring64,
    Ring128,
    Float32Tensor,
    Float64Tensor,
    Ring64Tensor,
    Ring128Tensor,
    BitTensor,
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
    ReplicatedSetup,
    Replicated64Tensor,
    Replicated128Tensor,
    ReplicatedBitTensor,
    Additive64Tensor,
    Additive128Tensor,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub enum Value {
    Unit,
    Shape(Shape),
    Seed(Seed),
    PrfKey(PrfKey),
    Nonce(Nonce),
    Float32(f32),
    Float64(f64),
    Ring64(u64),
    Ring128(u128),
    String(String),
    Ring64Tensor(Ring64Tensor),
    Ring128Tensor(Ring128Tensor),
    BitTensor(BitTensor),
    Float32Tensor(Float32Tensor),
    Float64Tensor(Float64Tensor),
    Int8Tensor(Int8Tensor),
    Int16Tensor(Int16Tensor),
    Int32Tensor(Int32Tensor),
    Int64Tensor(Int64Tensor),
    Uint8Tensor(Uint8Tensor),
    Uint16Tensor(Uint16Tensor),
    Uint32Tensor(Uint32Tensor),
    Uint64Tensor(Uint64Tensor),
    Fixed64Tensor(Fixed64Tensor),
    Fixed128Tensor(Fixed128Tensor),
    Replicated64Tensor(Replicated64Tensor),
    Replicated128Tensor(Replicated128Tensor),
    ReplicatedBitTensor(ReplicatedBitTensor),
    ReplicatedSetup(ReplicatedSetup),
    Additive64Tensor(Additive64Tensor),
    Additive128Tensor(Additive128Tensor),
}

impl Value {
    pub fn ty(&self) -> Ty {
        match self {
            Value::Unit => Ty::Unit,
            Value::String(_) => Ty::String,
            Value::Float32(_) => Ty::Float32,
            Value::Float64(_) => Ty::Float64,
            Value::Ring64(_) => Ty::Ring64,
            Value::Ring128(_) => Ty::Ring128,
            Value::Ring64Tensor(_) => Ty::Ring64Tensor,
            Value::Ring128Tensor(_) => Ty::Ring128Tensor,
            Value::BitTensor(_) => Ty::BitTensor,
            Value::Shape(_) => Ty::Shape,
            Value::Seed(_) => Ty::Seed,
            Value::PrfKey(_) => Ty::PrfKey,
            Value::Nonce(_) => Ty::Nonce,
            Value::Float32Tensor(_) => Ty::Float32Tensor,
            Value::Float64Tensor(_) => Ty::Float64Tensor,
            Value::Int8Tensor(_) => Ty::Int8Tensor,
            Value::Int16Tensor(_) => Ty::Int16Tensor,
            Value::Int32Tensor(_) => Ty::Int32Tensor,
            Value::Int64Tensor(_) => Ty::Int64Tensor,
            Value::Uint8Tensor(_) => Ty::Uint8Tensor,
            Value::Uint16Tensor(_) => Ty::Uint16Tensor,
            Value::Uint32Tensor(_) => Ty::Uint32Tensor,
            Value::Uint64Tensor(_) => Ty::Uint64Tensor,
            Value::Additive64Tensor(_) => Ty::Additive64Tensor,
            Value::Additive128Tensor(_) => Ty::Additive128Tensor,
            Value::Replicated64Tensor(_) => Ty::Replicated64Tensor,
            Value::Replicated128Tensor(_) => Ty::Replicated128Tensor,
            Value::ReplicatedBitTensor(_) => Ty::ReplicatedBitTensor,
            Value::ReplicatedSetup(_) => Ty::ReplicatedSetup,
            Value::Fixed64Tensor(_) => Ty::Fixed64Tensor,
            Value::Fixed128Tensor(_) => Ty::Fixed128Tensor,
        }
    }
}

pub trait KnownType {
    const TY: Ty;
}

macro_rules! value {
    ($raw_type:ident) => {
        impl From<$raw_type> for Value {
            fn from(x: $raw_type) -> Self {
                Value::$raw_type(x)
            }
        }

        impl TryFrom<Value> for $raw_type {
            type Error = Error;
            fn try_from(v: Value) -> Result<Self> {
                match v {
                    Value::$raw_type(x) => Ok(x),
                    _ => Err(Error::TypeMismatch {
                        expected: stringify!($raw_type).to_string(),
                        found: v.ty(),
                    }),
                }
            }
        }

        impl<'v> TryFrom<&'v Value> for &'v $raw_type {
            type Error = Error;
            fn try_from(v: &'v Value) -> Result<Self> {
                match v {
                    Value::$raw_type(x) => Ok(x),
                    _ => Err(Error::TypeMismatch {
                        expected: stringify!($raw_type).to_string(),
                        found: v.ty(),
                    }),
                }
            }
        }

        impl KnownType for $raw_type {
            const TY: Ty = Ty::$raw_type;
        }
    };
}

value!(String);
value!(Ring64Tensor);
value!(Ring128Tensor);
value!(BitTensor);
value!(Shape);
value!(Seed);
value!(PrfKey);
value!(Nonce);
value!(Float32Tensor);
value!(Float64Tensor);
value!(Int8Tensor);
value!(Int16Tensor);
value!(Int32Tensor);
value!(Int64Tensor);
value!(Uint8Tensor);
value!(Uint16Tensor);
value!(Uint32Tensor);
value!(Uint64Tensor);

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
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub enum Operator {
    IdentityOp(IdentityOp),
    LoadOp(LoadOp),
    SaveOp(SaveOp),
    SendOp(SendOp),
    ReceiveOp(ReceiveOp),
    InputOp(InputOp),
    OutputOp(OutputOp),
    ConstantOp(ConstantOp),
    StdAddOp(StdAddOp),
    StdSubOp(StdSubOp),
    StdMulOp(StdMulOp),
    StdDivOp(StdDivOp),
    StdDotOp(StdDotOp),
    StdMeanOp(StdMeanOp),
    StdExpandDimsOp(StdExpandDimsOp),
    StdReshapeOp(StdReshapeOp),
    StdAtLeast2DOp(StdAtLeast2DOp),
    StdShapeOp(StdShapeOp),
    StdSliceOp(StdSliceOp),
    StdSumOp(StdSumOp),
    StdOnesOp(StdOnesOp),
    StdConcatenateOp(StdConcatenateOp),
    StdTransposeOp(StdTransposeOp),
    StdInverseOp(StdInverseOp),
    RingAddOp(RingAddOp),
    RingSubOp(RingSubOp),
    RingMulOp(RingMulOp),
    RingDotOp(RingDotOp),
    RingSumOp(RingSumOp),
    RingShapeOp(RingShapeOp),
    RingSampleOp(RingSampleOp),
    RingFillOp(RingFillOp),
    RingShlOp(RingShlOp),
    RingShrOp(RingShrOp),
    RingInjectOp(RingInjectOp),
    BitExtractOp(BitExtractOp),
    BitSampleOp(BitSampleOp),
    BitFillOp(BitFillOp),
    BitXorOp(BitXorOp),
    BitAndOp(BitAndOp),
    PrimDeriveSeedOp(PrimDeriveSeedOp),
    PrimGenPrfKeyOp(PrimGenPrfKeyOp),
    FixedAddOp(FixedAddOp),
    FixedMulOp(FixedMulOp),
    FixedpointRingEncodeOp(FixedpointRingEncodeOp),
    FixedpointRingDecodeOp(FixedpointRingDecodeOp),
    FixedpointRingMeanOp(FixedpointRingMeanOp),
    AdtRevealOp(AdtRevealOp),
    AdtAddOp(AdtAddOp),
    AdtMulOp(AdtMulOp),
    RepSetupOp(RepSetupOp),
    RepShareOp(RepShareOp),
    RepRevealOp(RepRevealOp),
    RepAddOp(RepAddOp),
    RepMulOp(RepMulOp),
    RepToAdtOp(RepToAdtOp),
}

macro_rules! operators {
    ($($t:ident),+) => {
        $(
        impl From<$t> for Operator {
            fn from(x: $t) -> Operator {
                Operator::$t(x)
            }
        }
        )+

        impl Operator {
            pub fn sig(&self) -> &Signature {
                match self {
                    $(Operator::$t(op) => &op.sig,)+
                }
            }
        }
    };
}

operators![
    IdentityOp,
    LoadOp,
    SaveOp,
    SendOp,
    ReceiveOp,
    InputOp,
    OutputOp,
    ConstantOp,
    StdAddOp,
    StdSubOp,
    StdMulOp,
    StdDivOp,
    StdDotOp,
    StdMeanOp,
    StdExpandDimsOp,
    StdReshapeOp,
    StdAtLeast2DOp,
    StdShapeOp,
    StdSliceOp,
    StdSumOp,
    StdOnesOp,
    StdConcatenateOp,
    StdTransposeOp,
    StdInverseOp,
    RingAddOp,
    RingSubOp,
    RingMulOp,
    RingDotOp,
    RingSumOp,
    RingShapeOp,
    RingSampleOp,
    RingFillOp,
    RingShlOp,
    RingShrOp,
    RingInjectOp,
    BitExtractOp,
    BitSampleOp,
    BitFillOp,
    BitXorOp,
    BitAndOp,
    PrimDeriveSeedOp,
    PrimGenPrfKeyOp,
    FixedAddOp,
    FixedMulOp,
    FixedpointRingEncodeOp,
    FixedpointRingDecodeOp,
    FixedpointRingMeanOp,
    AdtRevealOp,
    AdtAddOp,
    AdtMulOp,
    RepSetupOp,
    RepShareOp,
    RepRevealOp,
    RepAddOp,
    RepMulOp,
    RepToAdtOp
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
    pub value: Value, // TODO Box<Value> or Box inside Value?
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
    pub nonce: Nonce,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct PrimGenPrfKeyOp {
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
    pub value: Value,
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
    sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct AdtAddOp {
    sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct AdtMulOp {
    sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RepSetupOp {
    sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RepShareOp {
    sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RepRevealOp {
    sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RepAddOp {
    sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RepMulOp {
    sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct RepToAdtOp {
    sig: Signature,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub enum Placement {
    Host(HostPlacement),
    Replicated(ReplicatedPlacement),
}

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

impl From<HostPlacement> for Placement {
    fn from(plc: HostPlacement) -> Self {
        Placement::Host(plc)
    }
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct ReplicatedPlacement {
    pub owners: [Role; 3],
}

impl From<ReplicatedPlacement> for Placement {
    fn from(plc: ReplicatedPlacement) -> Self {
        Placement::Replicated(plc)
    }
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
