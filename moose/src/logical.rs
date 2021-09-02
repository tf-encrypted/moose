use crate::computation::{AtLeast2DOp, HasShortName, HostPlacement, KnownType, MeanOp, Placed, Placement, Signature, SymbolicType};
use crate::error::Result;
use crate::fixedpoint::{Fixed128Tensor, Fixed64Tensor};
use crate::floatingpoint::{Float32Tensor, Float64Tensor, FloatTensor};
use crate::host::HostFloat64Tensor;
use crate::kernels::{PlacementAdd, PlacementAtLeast2D, PlacementMean, PlacementStdMean, Session};
use crate::symbolic::Symbolic;
use macros::with_context;
use macros::ShortName;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T> {
    Fixed64(Fixed64T),
    Fixed128(Fixed128T),
    Float32(Float32T),
    Float64(Float64T),
}

pub type Tensor = AbstractTensor<Fixed64Tensor, Fixed128Tensor, Float32Tensor, Float64Tensor>;

impl<Fixed64T, Fixed128T, Float32T, Float64T> Placed
    for AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>
where
    Fixed64T: Placed,
    Fixed64T::Placement: Into<Placement>,
    Fixed128T: Placed,
    Fixed128T::Placement: Into<Placement>,
    Float32T: Placed,
    Float32T::Placement: Into<Placement>,
    Float64T: Placed,
    Float64T::Placement: Into<Placement>,
{
    type Placement = Placement;

    fn placement(&self) -> Result<Self::Placement> {
        match self {
            AbstractTensor::Fixed64(x) => Ok(x.placement()?.into()),
            AbstractTensor::Fixed128(x) => Ok(x.placement()?.into()),
            AbstractTensor::Float32(x) => Ok(x.placement()?.into()),
            AbstractTensor::Float64(x) => Ok(x.placement()?.into()),
        }
    }
}

impl SymbolicType for Tensor {
    type Type = Symbolic<
        AbstractTensor<
            <Fixed64Tensor as SymbolicType>::Type,
            <Fixed128Tensor as SymbolicType>::Type,
            <Float32Tensor as SymbolicType>::Type,
            <Float64Tensor as SymbolicType>::Type,
        >,
    >;
}

// NOTE(Morten) trying something new here by keeping op structs in dialect file

// #[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
// pub struct LogicalAddOp {
//     pub sig: Signature,
// }

// modelled!(PlacementAdd::add, Placement, (Tensor, Tensor) -> Tensor, LogicalAddOp);

// hybrid_kernel! {
//     LogicalAddOp,
//     [
//         (Placement, (Tensor, Tensor) -> Tensor => Self::kernel),
//     ]
// }

// impl LogicalAddOp {
//     fn kernel<S: Session>(
//         sess: &S,
//         plc: &Placement,
//         x: Tensor,
//         y: Tensor,
//     ) -> Tensor
//     where
//     Placement: PlacementAdd<>
//     {
//         // TODO(Morten) would like to avoid with matching (since one is already done by `hybrid_kernel!`)
//         match (x, y) {
//             (Tensor::Fixed64(v), Tensor::Fixed64(w)) => {
//                 let result = with_context!(plc, sess, x + y);
//                 Tensor::Fixed64(result)
//             },
//             (Tensor::Fixed128(v), Tensor::Fixed128(w)) => {
//                 let result = with_context!(plc, sess, x + y);
//                 Tensor::Fixed128(result)
//             },
//             _ => unimplemented!() // TODO
//         }
//     }
// }

// #[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
// pub struct LogicalSubOp {
//     pub sig: Signature,
// }

// #[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
// pub struct LogicalMulOp {
//     pub sig: Signature,
// }

// #[derive(Serialize, Deserialize, PartialEq, Clone, Debug, ShortName)]
// pub struct LogicalDotOp {
//     pub sig: Signature,
// }

impl<Fixed64T, Fixed128T, Float32T, Float64T>
    From<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>>
    for Symbolic<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>>
where
    Fixed64T: Placed<Placement = Placement>,
    Fixed128T: Placed<Placement = Placement>,
    Float32T: Placed<Placement = Placement>,
    Float64T: Placed<Placement = Placement>,
{
    fn from(x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>) -> Self {
        Symbolic::Concrete(x)
    }
}

impl<Fixed64T, Fixed128T, Float32T, Float64T>
    TryFrom<Symbolic<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>>>
    for AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>
where
    Fixed64T: Placed<Placement = Placement>,
    Fixed128T: Placed<Placement = Placement>,
    Float32T: Placed<Placement = Placement>,
    Float64T: Placed<Placement = Placement>,
{
    type Error = ();
    fn try_from(
        v: Symbolic<AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>>,
    ) -> std::result::Result<Self, ()> {
        match v {
            Symbolic::Concrete(x) => Ok(x),
            _ => Err(()),
        }
    }
}

kernel! {
    AtLeast2DOp, [
        (HostPlacement, (Tensor) -> Tensor => [hybrid] attributes[to_column_vector] Self::kernel),
    ]
}

impl AtLeast2DOp {
    fn kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T>(
        sess: &S,
        plc: &HostPlacement,
        to_column_vector: bool,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
    ) -> AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>
    where
        // HostPlacement: PlacementAtLeast2D<S, Fixed64T, Fixed64T>,
        // HostPlacement: PlacementAtLeast2D<S, Fixed128T, Fixed128T>,
        HostPlacement: PlacementAtLeast2D<S, Float32T, Float32T>,
        HostPlacement: PlacementAtLeast2D<S, Float64T, Float64T>,
    {
        match x {
            // AbstractTensor::Fixed64(x) => {
            //     let z = plc.at_least_2d(sess, to_column_vector, &x);
            //     AbstractTensor::Fixed64(z)
            // }
            // AbstractTensor::Fixed128(x) => {
            //     let z = plc.at_least_2d(sess, to_column_vector, &x);
            //     AbstractTensor::Fixed128(z)
            // }
            AbstractTensor::Float32(x) => {
                let z = plc.at_least_2d(sess, to_column_vector, &x);
                AbstractTensor::Float32(z)
            }
            AbstractTensor::Float64(x) => {
                let z = plc.at_least_2d(sess, to_column_vector, &x);
                AbstractTensor::Float64(z)
            }
            _ => unimplemented!("Fill other match arms please"),
        }
    }
}

kernel! {
    MeanOp, [
        (HostPlacement, (Tensor) -> Tensor => [hybrid] attributes[axis] Self::kernel),
    ]
}

impl MeanOp {
    fn kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
    ) -> AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>
    where
        // HostPlacement: PlacementAtLeast2D<S, Fixed64T, Fixed64T>,
        // HostPlacement: PlacementAtLeast2D<S, Fixed128T, Fixed128T>,
        HostPlacement: PlacementStdMean<S, Float32T, Float32T>,
        HostPlacement: PlacementStdMean<S, Float64T, Float64T>,
    {
        match x {
            AbstractTensor::Float32(x) => {
                let z = plc.std_mean(sess, axis, &x);
                AbstractTensor::Float32(z)
            }
            AbstractTensor::Float64(x) => {
                let z = plc.std_mean(sess, axis, &x);
                AbstractTensor::Float64(z)
            }
            _ => unimplemented!("Fill other match arms please"),
        }
    }
}
