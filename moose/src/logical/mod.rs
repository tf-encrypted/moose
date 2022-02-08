//! Abstraction layer for high-level logical tensors

use crate::computation::{HasShortName, PartiallySymbolicType, Placed, Placement, SymbolicType};
use crate::error::Result;
use crate::execution::symbolic::Symbolic;
use crate::host::HostShape;
use crate::types::*;
use derive_more::Display;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

mod ops;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum Shape {
    Host(HostShape),
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Copy, Clone, Debug, Display)]
pub enum TensorDType {
    #[display(fmt = "Fixed64({}, {})", integral_precision, fractional_precision)]
    Fixed64 {
        integral_precision: u32,
        fractional_precision: u32,
    },
    #[display(fmt = "Fixed128({}, {})", integral_precision, fractional_precision)]
    Fixed128 {
        integral_precision: u32,
        fractional_precision: u32,
    },
    Float32,
    Float64,
    Bool,
    U64,
    Unknown,
}

impl HasShortName for TensorDType {
    fn short_name(&self) -> &str {
        match self {
            TensorDType::Fixed64 { .. } => "Fixed64",
            TensorDType::Fixed128 { .. } => "Fixed128",
            TensorDType::Float32 => "Float32",
            TensorDType::Float64 => "Float64",
            TensorDType::Bool => "Bool",
            TensorDType::U64 => "U64",
            TensorDType::Unknown => "Unknown",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, U64T> {
    Fixed64(Fixed64T),
    Fixed128(Fixed128T),
    Float32(Float32T),
    Float64(Float64T),
    Bool(BoolT),
    U64(U64T),
}

impl<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, U64T>
    AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, U64T>
{
    pub fn ty_desc(&self) -> String {
        match self {
            AbstractTensor::Fixed64(_) => "Tensor(Fixed64)",
            AbstractTensor::Fixed128(_) => "Tensor(Fixed128)",
            AbstractTensor::Float32(_) => "Tensor(Float32)",
            AbstractTensor::Float64(_) => "Tensor(Float64)",
            AbstractTensor::Bool(_) => "Tensor(Bool)",
            AbstractTensor::U64(_) => "Tensor(U64T)",
        }
        .to_string()
    }
}

impl<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, U64T> Placed
    for AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, U64T>
where
    Fixed64T: Placed,
    Fixed64T::Placement: Into<Placement>,
    Fixed128T: Placed,
    Fixed128T::Placement: Into<Placement>,
    Float32T: Placed,
    Float32T::Placement: Into<Placement>,
    Float64T: Placed,
    Float64T::Placement: Into<Placement>,
    BoolT: Placed,
    BoolT::Placement: Into<Placement>,
    U64T: Placed,
    U64T::Placement: Into<Placement>,
{
    type Placement = Placement;

    fn placement(&self) -> Result<Self::Placement> {
        match self {
            AbstractTensor::Fixed64(x) => Ok(x.placement()?.into()),
            AbstractTensor::Fixed128(x) => Ok(x.placement()?.into()),
            AbstractTensor::Float32(x) => Ok(x.placement()?.into()),
            AbstractTensor::Float64(x) => Ok(x.placement()?.into()),
            AbstractTensor::Bool(x) => Ok(x.placement()?.into()),
            AbstractTensor::U64(x) => Ok(x.placement()?.into()),
        }
    }
}

impl PartiallySymbolicType for Tensor {
    #[allow(clippy::type_complexity)]
    type Type = AbstractTensor<
        <Fixed64Tensor as SymbolicType>::Type,
        <Fixed128Tensor as SymbolicType>::Type,
        <Float32Tensor as SymbolicType>::Type,
        <Float64Tensor as SymbolicType>::Type,
        <BooleanTensor as SymbolicType>::Type,
        <Uint64Tensor as SymbolicType>::Type,
    >;
}

impl<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, U64T>
    From<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, U64T>>
    for Symbolic<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, U64T>>
where
    Fixed64T: Placed<Placement = Placement>,
    Fixed128T: Placed<Placement = Placement>,
    Float32T: Placed<Placement = Placement>,
    Float64T: Placed<Placement = Placement>,
    BoolT: Placed<Placement = Placement>,
    U64T: Placed<Placement = Placement>,
{
    fn from(x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, U64T>) -> Self {
        Symbolic::Concrete(x)
    }
}

impl<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, U64T>
    TryFrom<Symbolic<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, U64T>>>
    for AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, U64T>
where
    Fixed64T: Placed<Placement = Placement>,
    Fixed128T: Placed<Placement = Placement>,
    Float32T: Placed<Placement = Placement>,
    Float64T: Placed<Placement = Placement>,
    BoolT: Placed<Placement = Placement>,
    U64T: Placed<Placement = Placement>,
{
    type Error = ();
    fn try_from(
        v: Symbolic<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T, BoolT, U64T>>,
    ) -> std::result::Result<Self, ()> {
        match v {
            Symbolic::Concrete(x) => Ok(x),
            _ => Err(()),
        }
    }
}
