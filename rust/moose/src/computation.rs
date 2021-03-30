use crate::error::{Error, Result};
use crate::prim::{Nonce, PrfKey, Seed};
use crate::ring::{Ring128Tensor, Ring64Tensor};
use crate::standard::{
    Float32Tensor, Float64Tensor, Int16Tensor, Int32Tensor, Int64Tensor, Int8Tensor, Shape,
    Uint16Tensor, Uint32Tensor, Uint64Tensor, Uint8Tensor,
};
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

pub type RendezvousKey = str;

pub type SessionId = u128;

#[derive(Serialize, Deserialize, PartialEq, Copy, Clone, Debug)]
pub enum Ty {
    UnitTy,
    StringTy,
    Ring64TensorTy,
    Ring128TensorTy,
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
    Ring64Tensor(Ring64Tensor),
    Ring128Tensor(Ring128Tensor),
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
            Ring64Tensor(_) => Ring64TensorTy,
            Ring128Tensor(_) => Ring128TensorTy,
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

macro_rules! convert {
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
                    _ => Err(Error::TypeMismatch),
                }
            }
        }

        impl<'v> TryFrom<&'v Value> for &'v $raw_type {
            type Error = Error;
            fn try_from(v: &'v Value) -> Result<Self> {
                match v {
                    Value::$raw_type(x) => Ok(x),
                    _ => Err(Error::TypeMismatch),
                }
            }
        }
    };
}

convert!(Ring64Tensor);
convert!(Ring128Tensor);
convert!(Shape);
convert!(Seed);
convert!(PrfKey);
convert!(Nonce);
convert!(Float32Tensor);
convert!(Float64Tensor);
convert!(Int8Tensor);
convert!(Int16Tensor);
convert!(Int32Tensor);
convert!(Int64Tensor);
convert!(Uint8Tensor);
convert!(Uint16Tensor);
convert!(Uint32Tensor);
convert!(Uint64Tensor);

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Operator {
    Identity(IdentityOp),
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
    StdReshape(StdReshapeOp),
    StdSum(StdSumOp),
    StdOnes(StdOnesOp),
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
    PrimDeriveSeed(PrimDeriveSeedOp),
    PrimGenPrfKey(PrimGenPrfKeyOp),
    FixedpointRingEncode(FixedpointRingEncodeOp),
    FixedpointRingDecode(FixedpointRingDecodeOp),
    FixedpointRingMean(FixedpointRingMeanOp),
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SendOp {
    pub rendezvous_key: String,
    pub sender: HostPlacement,
    pub receiver: HostPlacement,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct IdentityOp {
    pub ty: Ty,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ReceiveOp {
    pub rendezvous_key: String,
    pub sender: HostPlacement,
    pub receiver: HostPlacement,
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
pub struct StdReshapeOp {
    pub ty: Ty,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StdSumOp {
    pub ty: Ty,
    pub axis: Option<u32>,
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

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HostPlacement {
    pub name: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ReplicatedPlacement {
    pub players: [String; 3],
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
