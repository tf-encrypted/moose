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
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Debug)]
pub enum Value {
    Unit,
    Float32(f32),
    Float64(f64),
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

#[derive(Serialize, Deserialize, Clone, Debug)]
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

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SendOp {
    pub rendezvous_key: String,
    pub receiver: Role,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct IdentityOp {
    pub ty: Ty,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ReceiveOp {
    pub rendezvous_key: String,
    pub sender: Role,
    pub ty: Ty,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct InputOp {
    pub arg_name: String,
    pub ty: Ty,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OutputOp {
    pub ty: Ty,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LoadOp {
    pub ty: Ty,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SaveOp {
    pub ty: Ty,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ConstantOp {
    pub value: Value,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StdAddOp {
    pub lhs: Ty,
    pub rhs: Ty,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StdSubOp {
    pub lhs: Ty,
    pub rhs: Ty,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StdMulOp {
    pub lhs: Ty,
    pub rhs: Ty,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StdDivOp {
    pub lhs: Ty,
    pub rhs: Ty,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StdDotOp {
    pub lhs: Ty,
    pub rhs: Ty,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StdMeanOp {
    pub ty: Ty,
    pub axis: Option<u32>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StdOnesOp {
    pub ty: Ty,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StdConcatenateOp {
    pub ty: Ty,
    pub axis: u32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StdAtLeast2DOp {
    pub ty: Ty,
    pub to_column_vector: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StdExpandDimsOp {
    pub ty: Ty,
    pub axis: u32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StdReshapeOp {
    pub ty: Ty,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StdShapeOp {
    pub ty: Ty,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StdSliceOp {
    pub ty: Ty,
    pub start: u32,
    pub end: u32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StdSumOp {
    pub ty: Ty,
    pub axis: Option<u32>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StdTransposeOp {
    pub ty: Ty,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StdInverseOp {
    pub ty: Ty,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PrimDeriveSeedOp {
    pub nonce: Nonce,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PrimGenPrfKeyOp;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingAddOp {
    pub lhs: Ty,
    pub rhs: Ty,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingSubOp {
    pub lhs: Ty,
    pub rhs: Ty,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingMulOp {
    pub lhs: Ty,
    pub rhs: Ty,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingDotOp {
    pub lhs: Ty,
    pub rhs: Ty,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingSumOp {
    pub ty: Ty,
    pub axis: Option<u32>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingShapeOp {
    pub ty: Ty,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingFillOp {
    pub value: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingSampleOp {
    pub output: Ty,
    pub max_value: Option<u64>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingShlOp {
    pub amount: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingShrOp {
    pub amount: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RingInjectOp {
    pub output: Ty,
    pub bit_idx: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BitExtractOp {
    pub ring_type: Ty,
    pub bit_idx: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BitSampleOp;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BitFillOp {
    pub value: u8,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BitXorOp;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BitAndOp;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FixedpointRingEncodeOp {
    pub scaling_factor: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FixedpointRingDecodeOp {
    pub scaling_factor: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FixedpointRingMeanOp {
    pub axis: Option<usize>,
    pub scaling_factor: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
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

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HostPlacement {
    pub owner: Role,
}

impl From<HostPlacement> for Placement {
    fn from(plc: HostPlacement) -> Self {
        Placement::Host(plc)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ReplicatedPlacement {
    pub owners: [Role; 3],
}

impl From<ReplicatedPlacement> for Placement {
    fn from(plc: ReplicatedPlacement) -> Self {
        Placement::Replicated(plc)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Operation {
    pub name: String,
    pub kind: Operator,
    pub inputs: Vec<String>, // TODO(Morten) use indices instead of strings?
    pub placement: Placement,
}

pub struct Computation {
    // pub constants: Vec<Value>,
    // pub operators: Vec<Operator>,
    pub operations: Vec<Operation>,
}
