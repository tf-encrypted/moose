use crate::bit::BitTensor;
use crate::error::{Error, Result};
use crate::prim::{Nonce, PrfKey, Seed};
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
    UnitTy,
    StringTy,
    Float32Ty,
    Float64Ty,
    Ring64Ty,
    Ring128Ty,
    Ring64TensorTy,
    Ring128TensorTy,
    BitTensorTy,
    ShapeTy,
    SeedTy,
    PrfKeyTy,
    NonceTy,
    Float32TensorTy,
    Float64TensorTy,
    Int8TensorTy,
    Int16TensorTy,
    Int32TensorTy,
    Int64TensorTy,
    Uint8TensorTy,
    Uint16TensorTy,
    Uint32TensorTy,
    Uint64TensorTy,
    UnknownTy,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub enum Value {
    Unit,
    Float32(f32),
    Float64(f64),
    Ring64(u64),
    Ring128(u128),
    String(String),
    Ring64Tensor(Ring64Tensor),
    Ring128Tensor(Ring128Tensor),
    BitTensor(BitTensor),
    Shape(Shape),
    Seed(Seed),
    PrfKey(PrfKey),
    Nonce(Nonce),
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
}

impl Value {
    pub fn ty(&self) -> Ty {
        use Ty::*;
        use Value::*;
        match self {
            Unit => UnitTy,
            String(_) => StringTy,
            Float32(_) => Float32Ty,
            Float64(_) => Float64Ty,
            Ring64(_) => Ring64Ty,
            Ring128(_) => Ring128Ty,
            Ring64Tensor(_) => Ring64TensorTy,
            Ring128Tensor(_) => Ring128TensorTy,
            BitTensor(_) => BitTensorTy,
            Shape(_) => ShapeTy,
            Seed(_) => SeedTy,
            PrfKey(_) => PrfKeyTy,
            Nonce(_) => NonceTy,
            Float32Tensor(_) => Float32TensorTy,
            Float64Tensor(_) => Float64TensorTy,
            Int8Tensor(_) => Int8TensorTy,
            Int16Tensor(_) => Int16TensorTy,
            Int32Tensor(_) => Int32TensorTy,
            Int64Tensor(_) => Int64TensorTy,
            Uint8Tensor(_) => Uint8TensorTy,
            Uint16Tensor(_) => Uint16TensorTy,
            Uint32Tensor(_) => Uint32TensorTy,
            Uint64Tensor(_) => Uint64TensorTy,
        }
    }
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

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub enum Signature {
    Nullary(NullarySignature),
    Unary(UnarySignature),
    Binary(BinarySignature),
    Ternary(TernarySignature),
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct NullarySignature {
    pub ret: Ty,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct UnarySignature {
    pub arg0: Ty,
    pub ret: Ty,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub struct BinarySignature {
    pub arg0: Ty,
    pub arg1: Ty,
    pub ret: Ty,
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
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
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub enum Operator {
    Identity(IdentityOp),
    Load(LoadOp),
    Save(SaveOp),
    Send(SendOp),
    Receive(ReceiveOp),
    Input(InputOp),
    Output(OutputOp),
    Constant(ConstantOp),
    StdAdd(StdAddOp),
    StdSub(StdSubOp),
    StdMul(StdMulOp),
    StdDiv(StdDivOp),
    StdDot(StdDotOp),
    StdMean(StdMeanOp),
    StdExpandDims(StdExpandDimsOp),
    StdReshape(StdReshapeOp),
    StdAtLeast2D(StdAtLeast2DOp),
    StdShape(StdShapeOp),
    StdSlice(StdSliceOp),
    StdSum(StdSumOp),
    StdOnes(StdOnesOp),
    StdConcatenate(StdConcatenateOp),
    StdTranspose(StdTransposeOp),
    StdInverse(StdInverseOp),
    RingAdd(RingAddOp),
    RingSub(RingSubOp),
    RingMul(RingMulOp),
    RingDot(RingDotOp),
    RingSum(RingSumOp),
    RingShape(RingShapeOp),
    RingSample(RingSampleOp),
    RingFill(RingFillOp),
    RingShl(RingShlOp),
    RingShr(RingShrOp),
    RingInject(RingInjectOp),
    BitExtract(BitExtractOp),
    BitSample(BitSampleOp),
    BitFill(BitFillOp),
    BitXor(BitXorOp),
    BitAnd(BitAndOp),
    PrimDeriveSeed(PrimDeriveSeedOp),
    PrimGenPrfKey(PrimGenPrfKeyOp),
    FixedpointRingEncode(FixedpointRingEncodeOp),
    FixedpointRingDecode(FixedpointRingDecodeOp),
    FixedpointRingMean(FixedpointRingMeanOp),
}

impl Operator {
    pub fn sig(&self) -> &Signature {
        match self {
            Operator::Identity(op) => &op.sig,
            Operator::Load(op) => &op.sig,
            Operator::Save(op) => &op.sig,
            Operator::Send(op) => &op.sig,
            Operator::Receive(op) => &op.sig,
            Operator::Input(op) => &op.sig,
            Operator::Output(op) => &op.sig,
            Operator::Constant(op) => &op.sig,
            Operator::StdAdd(op) => &op.sig,
            Operator::StdSub(op) => &op.sig,
            Operator::StdMul(op) => &op.sig,
            Operator::StdDiv(op) => &op.sig,
            Operator::StdDot(op) => &op.sig,
            Operator::StdMean(op) => &op.sig,
            Operator::StdExpandDims(op) => &op.sig,
            Operator::StdReshape(op) => &op.sig,
            Operator::StdAtLeast2D(op) => &op.sig,
            Operator::StdShape(op) => &op.sig,
            Operator::StdSlice(op) => &op.sig,
            Operator::StdSum(op) => &op.sig,
            Operator::StdOnes(op) => &op.sig,
            Operator::StdConcatenate(op) => &op.sig,
            Operator::StdTranspose(op) => &op.sig,
            Operator::StdInverse(op) => &op.sig,
            Operator::RingAdd(op) => &op.sig,
            Operator::RingSub(op) => &op.sig,
            Operator::RingMul(op) => &op.sig,
            Operator::RingDot(op) => &op.sig,
            Operator::RingSum(op) => &op.sig,
            Operator::RingShape(op) => &op.sig,
            Operator::RingSample(op) => &op.sig,
            Operator::RingFill(op) => &op.sig,
            Operator::RingShl(op) => &op.sig,
            Operator::RingShr(op) => &op.sig,
            Operator::RingInject(op) => &op.sig,
            Operator::BitExtract(op) => &op.sig,
            Operator::BitSample(op) => &op.sig,
            Operator::BitFill(op) => &op.sig,
            Operator::BitXor(op) => &op.sig,
            Operator::BitAnd(op) => &op.sig,
            Operator::PrimDeriveSeed(op) => &op.sig,
            Operator::PrimGenPrfKey(op) => &op.sig,
            Operator::FixedpointRingEncode(op) => &op.sig,
            Operator::FixedpointRingDecode(op) => &op.sig,
            Operator::FixedpointRingMean(op) => &op.sig,
        }
    }
}

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
    pub value: Value,
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
