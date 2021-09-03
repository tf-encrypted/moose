use crate::computation::{
    AtLeast2DOp, HasShortName, HostPlacement, KnownType, MeanOp, Placed, Placement, Signature,
    SymbolicType, ReplicatedPlacement, AddOp, SubOp,
};
use crate::error::Result;
use crate::fixedpoint::{Fixed128Tensor, Fixed64Tensor};
use crate::floatingpoint::{Float32Tensor, Float64Tensor, FloatTensor};
use crate::host::{HostFixed64Tensor, HostFloat64Tensor, HostShape};
use crate::kernels::{PlacementAdd, PlacementSub, PlacementAtLeast2D, PlacementMean, Session};
use crate::symbolic::Symbolic;
use macros::with_context;
use macros::ShortName;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum Shape {
    Host(HostShape),
}

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

modelled!(PlacementAdd::add, HostPlacement, (Tensor, Tensor) -> Tensor, AddOp);

macro_rules! logical_kernel {
    
}


kernel! {
    AddOp,
    [
        (HostPlacement, (Tensor, Tensor) -> Tensor => [hybrid] Self::host_kernel),
        (ReplicatedPlacement, (Tensor, Tensor) -> Tensor => [hybrid] Self::rep_kernel),
    ]
}

impl AddOp {
    fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
    ) -> AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>
    where
        HostPlacement: PlacementAdd<S, Fixed64T, Fixed64T, Fixed64T>,
        HostPlacement: PlacementAdd<S, Fixed128T, Fixed128T, Fixed128T>,
        // HostPlacement: PlacementAdd<S, Float32T, Float32T, Float32T>,
        // HostPlacement: PlacementAdd<S, Float64T, Float64T, Float64T>,
    {
        match (x, y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let result = plc.add(sess, &x, &y);
                AbstractTensor::Fixed64(result)
            },
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let result = plc.add(sess, &x, &y);
                AbstractTensor::Fixed128(result)
            },
            // (AbstractTensor::Float32(x), AbstractTensor::Float32(y)) => {
            //     let result = plc.add(sess, &x, &y);
            //     AbstractTensor::Float32(result)
            // },
            // (AbstractTensor::Float64(x), AbstractTensor::Float64(y)) => {
            //     let result = plc.add(sess, &x, &y);
            //     AbstractTensor::Float64(result)
            // },
            _ => unimplemented!() // TOD(Morten) would be nice to catch statically; perhaps if custom kernel?!
        }
    }

    fn rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
    ) -> AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>
    where
        ReplicatedPlacement: PlacementAdd<S, Fixed64T, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementAdd<S, Fixed128T, Fixed128T, Fixed128T>,
        // HostPlacement: PlacementAdd<S, Float32T, Float32T, Float32T>,
        // HostPlacement: PlacementAdd<S, Float64T, Float64T, Float64T>,
    {
        match (x, y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let result = plc.add(sess, &x, &y);
                AbstractTensor::Fixed64(result)
            },
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let result = plc.add(sess, &x, &y);
                AbstractTensor::Fixed128(result)
            },
            // (AbstractTensor::Float32(x), AbstractTensor::Float32(y)) => {
            //     let result = plc.add(sess, &x, &y);
            //     AbstractTensor::Float32(result)
            // },
            // (AbstractTensor::Float64(x), AbstractTensor::Float64(y)) => {
            //     let result = plc.add(sess, &x, &y);
            //     AbstractTensor::Float64(result)
            // },
            _ => unimplemented!() // TOD(Morten) would be nice to catch statically; perhaps if custom kernel?!
        }
    }
}

modelled!(PlacementSub::sub, HostPlacement, (Tensor, Tensor) -> Tensor, SubOp);

kernel! {
    SubOp,
    [
        (HostPlacement, (Tensor, Tensor) -> Tensor => [hybrid] Self::host_kernel),
    ]
}

impl SubOp {
    fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
    ) -> AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>
    where
        HostPlacement: PlacementSub<S, Fixed64T, Fixed64T, Fixed64T>,
        HostPlacement: PlacementSub<S, Fixed128T, Fixed128T, Fixed128T>,
        // HostPlacement: PlacementAdd<S, Float32T, Float32T, Float32T>,
        // HostPlacement: PlacementAdd<S, Float64T, Float64T, Float64T>,
    {
        match (x, y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let result = plc.sub(sess, &x, &y);
                AbstractTensor::Fixed64(result)
            },
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let result = plc.sub(sess, &x, &y);
                AbstractTensor::Fixed128(result)
            },
            // (AbstractTensor::Float32(x), AbstractTensor::Float32(y)) => {
            //     let result = plc.add(sess, &x, &y);
            //     AbstractTensor::Float32(result)
            // },
            // (AbstractTensor::Float64(x), AbstractTensor::Float64(y)) => {
            //     let result = plc.add(sess, &x, &y);
            //     AbstractTensor::Float64(result)
            // },
            _ => unimplemented!() // TOD(Morten) would be nice to catch statically; perhaps if custom kernel?!
        }
    }
}

kernel! {
    AtLeast2DOp, [
        (HostPlacement, (Tensor) -> Tensor => [hybrid] attributes[to_column_vector] Self::host_kernel),
        // (ReplicatedPlacement, (Tensor) -> Tensor => [hybrid] attributes[to_column_vector] Self::rep_kernel),
    ]
}

impl AtLeast2DOp {
    fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T>(
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
        // TODO(Morten) could we use a single entry for Placement instead?
        (HostPlacement, (Tensor) -> Tensor => [hybrid] attributes[axis] Self::host_kernel),
        (ReplicatedPlacement, (Tensor) -> Tensor => [hybrid] attributes[axis] Self::rep_kernel),
    ]
}

impl MeanOp {
    fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
    ) -> AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>
    where
        HostPlacement: PlacementMean<S, Fixed64T, Fixed64T>,
        HostPlacement: PlacementMean<S, Fixed128T, Fixed128T>,
        HostPlacement: PlacementMean<S, Float32T, Float32T>,
        HostPlacement: PlacementMean<S, Float64T, Float64T>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                let z = plc.mean(sess, axis, &x);
                AbstractTensor::Fixed64(z)
            }
            AbstractTensor::Fixed128(x) => {
                let z = plc.mean(sess, axis, &x);
                AbstractTensor::Fixed128(z)
            }
            AbstractTensor::Float32(x) => {
                let z = plc.mean(sess, axis, &x);
                AbstractTensor::Float32(z)
            }
            AbstractTensor::Float64(x) => {
                let z = plc.mean(sess, axis, &x);
                AbstractTensor::Float64(z)
            }
        }
    }

    fn rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: Option<u32>,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
    ) -> AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>
    where
        ReplicatedPlacement: PlacementMean<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementMean<S, Fixed128T, Fixed128T>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                let z = plc.mean(sess, axis, &x);
                AbstractTensor::Fixed64(z)
            }
            AbstractTensor::Fixed128(x) => {
                let z = plc.mean(sess, axis, &x);
                AbstractTensor::Fixed128(z)
            }
            // TODO(Morten) the fact that the following two are unimplemented
            // would be nice to know at (Moose) compile time
            AbstractTensor::Float32(_x) => {
                unimplemented!()
            }
            AbstractTensor::Float64(_x) => {
                unimplemented!()
            }
        }
    }
}
