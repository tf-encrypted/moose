use crate::computation::{AddOp, AtLeast2DOp, CastOp, ConcatOp, DivOp, DotOp, ExpandDimsOp, HostPlacement, InverseOp, KnownType, LoadOp, MeanOp, MulOp, OnesOp, Placed, Placement, ReplicatedPlacement, ShapeOp, Signature, SubOp, SumOp, SymbolicType, TransposeOp};
use crate::error::Result;
use crate::fixedpoint::{Fixed128Tensor, Fixed64Tensor};
use crate::floatingpoint::{Float32Tensor, Float64Tensor};
use crate::host::HostShape;
use crate::kernels::{PlacementAdd, PlacementAtLeast2D, PlacementCast, PlacementConcatenate, PlacementDiv, PlacementDot, PlacementExpandDims, PlacementFixedpointDecode, PlacementFixedpointEncode, PlacementInverse, PlacementLoad, PlacementMean, PlacementMul, PlacementOnes, PlacementShape, PlacementSub, PlacementSum, PlacementTranspose, PlacementTruncPr, Session};
use crate::symbolic::Symbolic;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

// TODO get rid of this
const FIXEDPOINT_PRECISON: u32 = 27;

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
        HostPlacement: PlacementAdd<S, Float32T, Float32T, Float32T>,
        HostPlacement: PlacementAdd<S, Float64T, Float64T, Float64T>,
    {
        match (x, y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let result = plc.add(sess, &x, &y);
                AbstractTensor::Fixed64(result)
            }
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let result = plc.add(sess, &x, &y);
                AbstractTensor::Fixed128(result)
            }
            (AbstractTensor::Float32(x), AbstractTensor::Float32(y)) => {
                let result = plc.add(sess, &x, &y);
                AbstractTensor::Float32(result)
            }
            (AbstractTensor::Float64(x), AbstractTensor::Float64(y)) => {
                let result = plc.add(sess, &x, &y);
                AbstractTensor::Float64(result)
            }
            _ => unimplemented!(), // TOD(Morten) would be nice to catch statically; perhaps if custom kernel?!
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
    {
        match (x, y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let result = plc.add(sess, &x, &y);
                AbstractTensor::Fixed64(result)
            }
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let result = plc.add(sess, &x, &y);
                AbstractTensor::Fixed128(result)
            }
            _ => unimplemented!(), // TOD(Morten) would be nice to catch statically; perhaps if custom kernel?!
        }
    }
}

modelled!(PlacementSub::sub, HostPlacement, (Tensor, Tensor) -> Tensor, SubOp);

kernel! {
    SubOp,
    [
        (HostPlacement, (Tensor, Tensor) -> Tensor => [hybrid] Self::host_kernel),
        (ReplicatedPlacement, (Tensor, Tensor) -> Tensor => [hybrid] Self::rep_kernel),
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
        HostPlacement: PlacementSub<S, Float32T, Float32T, Float32T>,
        HostPlacement: PlacementSub<S, Float64T, Float64T, Float64T>,
    {
        match (x, y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let result = plc.sub(sess, &x, &y);
                AbstractTensor::Fixed64(result)
            }
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let result = plc.sub(sess, &x, &y);
                AbstractTensor::Fixed128(result)
            }
            (AbstractTensor::Float32(x), AbstractTensor::Float32(y)) => {
                let result = plc.sub(sess, &x, &y);
                AbstractTensor::Float32(result)
            }
            (AbstractTensor::Float64(x), AbstractTensor::Float64(y)) => {
                let result = plc.sub(sess, &x, &y);
                AbstractTensor::Float64(result)
            }
            _ => unimplemented!(), // TOD(Morten) would be nice to catch statically; perhaps if custom kernel?!
        }
    }

    fn rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
    ) -> AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>
    where
        ReplicatedPlacement: PlacementSub<S, Fixed64T, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementSub<S, Fixed128T, Fixed128T, Fixed128T>,
    {
        match (x, y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let result = plc.sub(sess, &x, &y);
                AbstractTensor::Fixed64(result)
            }
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let result = plc.sub(sess, &x, &y);
                AbstractTensor::Fixed128(result)
            }
            _ => unimplemented!(), // TOD(Morten) would be nice to catch statically; perhaps if custom kernel?!
        }
    }
}

modelled!(PlacementMul::mul, HostPlacement, (Tensor, Tensor) -> Tensor, MulOp);

kernel! {
    MulOp,
    [
        (HostPlacement, (Tensor, Tensor) -> Tensor => [hybrid] Self::host_kernel),
        (ReplicatedPlacement, (Tensor, Tensor) -> Tensor => [hybrid] Self::rep_kernel),
    ]
}

impl MulOp {
    fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
    ) -> AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>
    where
        HostPlacement: PlacementMul<S, Fixed64T, Fixed64T, Fixed64T>,
        HostPlacement: PlacementMul<S, Fixed128T, Fixed128T, Fixed128T>,
        HostPlacement: PlacementTruncPr<S, Fixed64T, Fixed64T>,
        HostPlacement: PlacementTruncPr<S, Fixed128T, Fixed128T>,
        HostPlacement: PlacementMul<S, Float32T, Float32T, Float32T>,
        HostPlacement: PlacementMul<S, Float64T, Float64T, Float64T>,
    {
        match (x, y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let z = plc.mul(sess, &x, &y);
                let result = plc.trunc_pr(sess, FIXEDPOINT_PRECISON, &z);
                AbstractTensor::Fixed64(result)
            }
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let z = plc.mul(sess, &x, &y);
                let result = plc.trunc_pr(sess, FIXEDPOINT_PRECISON, &z);
                AbstractTensor::Fixed128(result)
            }
            (AbstractTensor::Float32(x), AbstractTensor::Float32(y)) => {
                let result = plc.mul(sess, &x, &y);
                AbstractTensor::Float32(result)
            }
            (AbstractTensor::Float64(x), AbstractTensor::Float64(y)) => {
                let result = plc.mul(sess, &x, &y);
                AbstractTensor::Float64(result)
            }
            _ => unimplemented!(), // TOD(Morten) would be nice to catch statically; perhaps if custom kernel?!
        }
    }

    fn rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
    ) -> AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>
    where
        ReplicatedPlacement: PlacementMul<S, Fixed64T, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementMul<S, Fixed128T, Fixed128T, Fixed128T>,
        ReplicatedPlacement: PlacementTruncPr<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementTruncPr<S, Fixed128T, Fixed128T>,
    {
        match (x, y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let z = plc.mul(sess, &x, &y);
                let result = plc.trunc_pr(sess, FIXEDPOINT_PRECISON, &z);
                AbstractTensor::Fixed64(result)
            }
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let z = plc.mul(sess, &x, &y);
                let result = plc.trunc_pr(sess, FIXEDPOINT_PRECISON, &z);
                AbstractTensor::Fixed128(result)
            }
            _ => unimplemented!(), // TOD(Morten) would be nice to catch statically; perhaps if custom kernel?!
        }
    }
}

modelled!(PlacementDiv::div, HostPlacement, (Tensor, Tensor) -> Tensor, DivOp);

kernel! {
    DivOp,
    [
        (HostPlacement, (Tensor, Tensor) -> Tensor => [hybrid] Self::host_kernel),
        // (ReplicatedPlacement, (Tensor, Tensor) -> Tensor => [hybrid] Self::rep_kernel),
    ]
}

impl DivOp {
    fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
    ) -> AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>
    where
        // HostPlacement: PlacementDiv<S, Fixed64T, Fixed64T, Fixed64T>,
        // HostPlacement: PlacementDiv<S, Fixed128T, Fixed128T, Fixed128T>,
        // HostPlacement: PlacementTruncPr<S, Fixed64T, Fixed64T>,
        // HostPlacement: PlacementTruncPr<S, Fixed128T, Fixed128T>,
        HostPlacement: PlacementDiv<S, Float32T, Float32T, Float32T>,
        HostPlacement: PlacementDiv<S, Float64T, Float64T, Float64T>,
    {
        match (x, y) {
            // TODO impl host fixed-point division
            // (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
            //     let z = plc.div(sess, &x, &y);
            //     let result = plc.trunc_pr(sess, FIXEDPOINT_PRECISON, &z);
            //     AbstractTensor::Fixed64(result)
            // }
            // (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
            //     let z = plc.div(sess, &x, &y);
            //     let result = plc.trunc_pr(sess, FIXEDPOINT_PRECISON, &z);
            //     AbstractTensor::Fixed128(result)
            // }
            (AbstractTensor::Float32(x), AbstractTensor::Float32(y)) => {
                let result = plc.div(sess, &x, &y);
                AbstractTensor::Float32(result)
            }
            (AbstractTensor::Float64(x), AbstractTensor::Float64(y)) => {
                let result = plc.div(sess, &x, &y);
                AbstractTensor::Float64(result)
            }
            _ => unimplemented!(), // TOD(Morten) would be nice to catch statically; perhaps if custom kernel?!
        }
    }

    fn rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
    ) -> AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T> {
        unimplemented!()
    }
}

modelled!(PlacementDot::dot, HostPlacement, (Tensor, Tensor) -> Tensor, DotOp);

kernel! {
    DotOp,
    [
        (HostPlacement, (Tensor, Tensor) -> Tensor => [hybrid] Self::host_kernel),
        (ReplicatedPlacement, (Tensor, Tensor) -> Tensor => [hybrid] Self::rep_kernel),
    ]
}

impl DotOp {
    fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
    ) -> AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>
    where
        HostPlacement: PlacementDot<S, Fixed64T, Fixed64T, Fixed64T>,
        HostPlacement: PlacementDot<S, Fixed128T, Fixed128T, Fixed128T>,
        HostPlacement: PlacementTruncPr<S, Fixed64T, Fixed64T>,
        HostPlacement: PlacementTruncPr<S, Fixed128T, Fixed128T>,
        HostPlacement: PlacementDot<S, Float32T, Float32T, Float32T>,
        HostPlacement: PlacementDot<S, Float64T, Float64T, Float64T>,
    {
        match (x, y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let z = plc.dot(sess, &x, &y);
                let result = plc.trunc_pr(sess, FIXEDPOINT_PRECISON, &z);
                AbstractTensor::Fixed64(result)
            }
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let z = plc.dot(sess, &x, &y);
                let result = plc.trunc_pr(sess, FIXEDPOINT_PRECISON, &z);
                AbstractTensor::Fixed128(result)
            }
            (AbstractTensor::Float32(x), AbstractTensor::Float32(y)) => {
                let result = plc.dot(sess, &x, &y);
                AbstractTensor::Float32(result)
            }
            (AbstractTensor::Float64(x), AbstractTensor::Float64(y)) => {
                let result = plc.dot(sess, &x, &y);
                AbstractTensor::Float64(result)
            }
            _ => unimplemented!(), // TOD(Morten) would be nice to catch statically; perhaps if custom kernel?!
        }
    }

    fn rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
    ) -> AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>
    where
        ReplicatedPlacement: PlacementDot<S, Fixed64T, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementDot<S, Fixed128T, Fixed128T, Fixed128T>,
        ReplicatedPlacement: PlacementTruncPr<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementTruncPr<S, Fixed128T, Fixed128T>,
    {
        match (x, y) {
            (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
                let z = plc.dot(sess, &x, &y);
                let result = plc.trunc_pr(sess, FIXEDPOINT_PRECISON, &z);
                AbstractTensor::Fixed64(result)
            }
            (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
                let z = plc.dot(sess, &x, &y);
                let result = plc.trunc_pr(sess, FIXEDPOINT_PRECISON, &z);
                AbstractTensor::Fixed128(result)
            }
            _ => unimplemented!(), // TOD(Morten) would be nice to catch statically; perhaps if custom kernel?!
        }
    }
}

modelled!(PlacementCast::cast, HostPlacement, (Tensor) -> Tensor, CastOp);

kernel! {
    CastOp,
    [
        (HostPlacement, (Tensor) -> Tensor => [hybrid] Self::kernel),
    ]
}

// TODO(Morten) right now we fix what you can cast to and from; we could
// perhaps use a `dtype` attribute to make this more flexible
impl CastOp {
    fn kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
    ) -> AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>
    where
        HostPlacement: PlacementFixedpointDecode<S, Fixed64T, Float32T>,
        HostPlacement: PlacementFixedpointDecode<S, Fixed128T, Float64T>,
        HostPlacement: PlacementFixedpointEncode<S, Float32T, Fixed64T>,
        HostPlacement: PlacementFixedpointEncode<S, Float64T, Fixed128T>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                let inner = plc.fixedpoint_decode(sess, FIXEDPOINT_PRECISON, &x);
                AbstractTensor::Float32(inner)
            }
            AbstractTensor::Fixed128(x) => {
                let inner = plc.fixedpoint_decode(sess, FIXEDPOINT_PRECISON, &x);
                AbstractTensor::Float64(inner)
            }
            AbstractTensor::Float32(x) => {
                let inner = plc.fixedpoint_encode(sess, FIXEDPOINT_PRECISON, &x);
                AbstractTensor::Fixed64(inner)
            }
            AbstractTensor::Float64(x) => {
                let inner = plc.fixedpoint_encode(sess, FIXEDPOINT_PRECISON, &x);
                AbstractTensor::Fixed128(inner)
            }
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
            AbstractTensor::Fixed64(x) => {
                unimplemented!()
                // let z = plc.at_least_2d(sess, to_column_vector, &x);
                // AbstractTensor::Fixed64(z)
            }
            AbstractTensor::Fixed128(x) => {
                unimplemented!()
                // let z = plc.at_least_2d(sess, to_column_vector, &x);
                // AbstractTensor::Fixed128(z)
            }
            AbstractTensor::Float32(x) => {
                let z = plc.at_least_2d(sess, to_column_vector, &x);
                AbstractTensor::Float32(z)
            }
            AbstractTensor::Float64(x) => {
                let z = plc.at_least_2d(sess, to_column_vector, &x);
                AbstractTensor::Float64(z)
            }
        }
    }
}

kernel! {
    MeanOp, [
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
        // HostPlacement: PlacementMean<S, Float32T, Float32T>,
        // HostPlacement: PlacementMean<S, Float64T, Float64T>,
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
                unimplemented!()
                // let z = plc.mean(sess, axis, &x);
                // AbstractTensor::Float32(z)
            }
            AbstractTensor::Float64(x) => {
                unimplemented!()
                // let z = plc.mean(sess, axis, &x);
                // AbstractTensor::Float64(z)
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
            // TODO(Morten) the fact that we are limited on replicated
            // placements  would be nice to know at (Moose) compile time
            _ => unimplemented!(),
        }
    }
}

kernel! {
    SumOp, [
        (HostPlacement, (Tensor) -> Tensor => [hybrid] attributes[axis] Self::host_kernel),
        (ReplicatedPlacement, (Tensor) -> Tensor => [hybrid] attributes[axis] Self::rep_kernel),
    ]
}

impl SumOp {
    fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
    ) -> AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>
    where
        HostPlacement: PlacementSum<S, Fixed64T, Fixed64T>,
        HostPlacement: PlacementSum<S, Fixed128T, Fixed128T>,
        // HostPlacement: PlacementSum<S, Float32T, Float32T>,
        // HostPlacement: PlacementSum<S, Float64T, Float64T>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                let z = plc.sum(sess, axis, &x);
                AbstractTensor::Fixed64(z)
            }
            AbstractTensor::Fixed128(x) => {
                let z = plc.sum(sess, axis, &x);
                AbstractTensor::Fixed128(z)
            }
            AbstractTensor::Float32(x) => {
                unimplemented!()
                // let z = plc.sum(sess, axis, &x);
                // AbstractTensor::Float32(z)
            }
            AbstractTensor::Float64(x) => {
                unimplemented!()
                // let z = plc.sum(sess, axis, &x);
                // AbstractTensor::Float64(z)
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
        ReplicatedPlacement: PlacementSum<S, Fixed64T, Fixed64T>,
        ReplicatedPlacement: PlacementSum<S, Fixed128T, Fixed128T>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                let z = plc.sum(sess, axis, &x);
                AbstractTensor::Fixed64(z)
            }
            AbstractTensor::Fixed128(x) => {
                let z = plc.sum(sess, axis, &x);
                AbstractTensor::Fixed128(z)
            }
            _ => unimplemented!(),
        }
    }
}

modelled!(PlacementOnes::ones, HostPlacement, (HostShape) -> Tensor, OnesOp);
// TODO(lvorona): figure out modelled op for the replicated tensor
// modelled!(PlacementOnes::ones, ReplicatedPlacement, (HostShape) -> Tensor, OnesOp);

kernel! {
    OnesOp,
    [
        (HostPlacement, (HostShape) -> Tensor => [hybrid] Self::host_kernel),
        // (ReplicatedPlacement, (HostShape) -> Tensor => [hybrid] Self::rep_kernel),
    ]
}

impl OnesOp {
    fn host_kernel<S: Session>(
        sess: &S,
        plc: &HostPlacement,
        shape: cs!(HostShape),
    ) -> AbstractTensor<
        cs!(Fixed64Tensor),
        cs!(Fixed128Tensor),
        cs!(Float32Tensor),
        cs!(Float64Tensor),
    >
    where
        HostShape: KnownType<S>,
        Fixed64Tensor: KnownType<S>,
        Fixed128Tensor: KnownType<S>,
        Float32Tensor: KnownType<S>,
        Float64Tensor: KnownType<S>,
        HostPlacement: PlacementOnes<S, cs!(HostShape), cs!(Float64Tensor)>,
    {
        let result = plc.ones(sess, &shape);
        AbstractTensor::Float64(result)
    }

    // fn rep_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T>(
    //     sess: &S,
    //     plc: &ReplicatedPlacement,
    //     shape: cs!(HostShape),
    // ) -> AbstractTensor<cs!(Fixed64Tensor), cs!(Fixed128Tensor), cs!(Float32Tensor), cs!(Float64Tensor)>
    // where
    //     HostShape: KnownType<S>,
    //     Fixed64Tensor: KnownType<S>,
    //     Fixed128Tensor: KnownType<S>,
    //     Float32Tensor: KnownType<S>,
    //     Float64Tensor: KnownType<S>,
    //     HostPlacement: PlacementOnes<S, cs!(HostShape), cs!(Fixed128Tensor)>,
    // {
    //     let result = plc.ones(sess, &shape);
    //     AbstractTensor::Fixed128(result)
    // }
}

modelled!(PlacementExpandDims::expand_dims, HostPlacement, attributes[axis: Vec<u32>] (Tensor) -> Tensor, ExpandDimsOp);

kernel! {
    ExpandDimsOp,
    [
        (HostPlacement, (Tensor) -> Tensor => [hybrid] attributes[axis] Self::host_kernel),
    ]
}

impl ExpandDimsOp {
    fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T>(
        sess: &S,
        plc: &HostPlacement,
        axis: Vec<u32>,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
    ) -> AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>
    where
        HostPlacement: PlacementExpandDims<S, Float32T, Float32T>,
        HostPlacement: PlacementExpandDims<S, Float64T, Float64T>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                unimplemented!()
                // let z = plc.expand_dims(sess, axis, &x);
                // AbstractTensor::Fixed64(z)
            }
            AbstractTensor::Fixed128(x) => {
                unimplemented!()
                // let z = plc.expand_dims(sess, axis, &x);
                // AbstractTensor::Fixed128(z)
            }
            AbstractTensor::Float32(x) => {
                let z = plc.expand_dims(sess, axis, &x);
                AbstractTensor::Float32(z)
            }
            AbstractTensor::Float64(x) => {
                let z = plc.expand_dims(sess, axis, &x);
                AbstractTensor::Float64(z)
            }
        }
    }
}

modelled!(PlacementConcatenate::concatenate, HostPlacement, attributes[axis: u32] (Tensor, Tensor) -> Tensor, ConcatOp);

kernel! {
    ConcatOp, [
        (HostPlacement, (Tensor, Tensor) -> Tensor => [hybrid] attributes[axis] Self::host_kernel),
    ]
}

impl ConcatOp {
    fn host_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T>(
        sess: &S,
        plc: &HostPlacement,
        axis: u32,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
        y: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
    ) -> AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>
    where
        HostPlacement: PlacementConcatenate<S, Float32T, Float32T, Float32T>,
        HostPlacement: PlacementConcatenate<S, Float64T, Float64T, Float64T>,
    {
        match (x, y) {
            // (AbstractTensor::Fixed64(x), AbstractTensor::Fixed64(y)) => {
            //     let result = plc.concatenate(sess, axis, &x, &y);
            //     AbstractTensor::Fixed64(result)
            // }
            // (AbstractTensor::Fixed128(x), AbstractTensor::Fixed128(y)) => {
            //     let result = plc.concatenate(sess, axis, &x, &y);
            //     AbstractTensor::Fixed128(result)
            // }
            (AbstractTensor::Float32(x), AbstractTensor::Float32(y)) => {
                let result = plc.concatenate(sess, axis, &x, &y);
                AbstractTensor::Float32(result)
            }
            (AbstractTensor::Float64(x), AbstractTensor::Float64(y)) => {
                let result = plc.concatenate(sess, axis, &x, &y);
                AbstractTensor::Float64(result)
            }
            _ => unimplemented!("ConcatOp missing an implementation"), // TOD(Morten) would be nice to catch statically; perhaps if custom kernel?!
        }
    }
}

modelled!(PlacementTranspose::transpose, HostPlacement, (Tensor) -> Tensor, TransposeOp);

kernel! {
    TransposeOp, [
        (HostPlacement, (Tensor) -> Tensor => [hybrid] Self::kernel),
    ]
}

impl TransposeOp {
    pub fn kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
    ) -> AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>
    where
        // HostPlacement: PlacementTranspose<S, Float32T, Float32T>,
        HostPlacement: PlacementTranspose<S, Float64T, Float64T>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                unimplemented!()
                // let z = plc.transpose(sess, &x);
                // AbstractTensor::Fixed64(z)
            }
            AbstractTensor::Fixed128(x) => {
                unimplemented!()
                // let z = plc.transpose(sess, &x);
                // AbstractTensor::Fixed128(z)
            }
            AbstractTensor::Float32(x) => {
                unimplemented!()
                // let z = plc.transpose(sess, &x);
                // AbstractTensor::Float32(z)
            }
            AbstractTensor::Float64(x) => {
                let z = plc.transpose(sess, &x);
                AbstractTensor::Float64(z)
            }
        }
    }
}

modelled!(PlacementInverse::inverse, HostPlacement, (Tensor) -> Tensor, InverseOp);

kernel! {
    InverseOp, [
        (HostPlacement, (Tensor) -> Tensor => [hybrid] Self::kernel),
    ]
}

impl InverseOp {
    pub fn kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
    ) -> AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>
    where
        // HostPlacement: PlacementInverse<S, Float32T, Float32T>,
        HostPlacement: PlacementInverse<S, Float64T, Float64T>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                unimplemented!()
                // let z = plc.inverse(sess, &x);
                // AbstractTensor::Fixed64(z)
            }
            AbstractTensor::Fixed128(x) => {
                unimplemented!()
                // let z = plc.inverse(sess, &x);
                // AbstractTensor::Fixed128(z)
            }
            AbstractTensor::Float32(x) => {
                unimplemented!()
                // let z = plc.inverse(sess, &x);
                // AbstractTensor::Float32(z)
            }
            AbstractTensor::Float64(x) => {
                let z = plc.inverse(sess, &x);
                AbstractTensor::Float64(z)
            }
        }
    }
}

impl LoadOp {
    pub fn logical_kernel<S: Session>(
        sess: &S,
        plc: &HostPlacement,
        key: cs!(String),
        query: cs!(String),
    ) -> AbstractTensor<
        cs!(Fixed64Tensor),
        cs!(Fixed128Tensor),
        cs!(Float32Tensor),
        cs!(Float64Tensor),
    >
    where
        String: KnownType<S>,
        Fixed64Tensor: KnownType<S>,
        Fixed128Tensor: KnownType<S>,
        Float32Tensor: KnownType<S>,
        Float64Tensor: KnownType<S>,
        HostPlacement: PlacementLoad<S, cs!(String), cs!(String), cs!(Float64Tensor)>,
    {
        let z = plc.load(sess, &key, &query);
        AbstractTensor::Float64(z)
    }
}

impl ShapeOp {
    pub(crate) fn logical_kernel<S: Session, Fixed64T, Fixed128T, Float32T, Float64T>(
        sess: &S,
        plc: &HostPlacement,
        x: AbstractTensor<Fixed64T, Fixed128T, Float32T, Float64T>,
    ) -> cs!(HostShape)
    where
        HostShape: KnownType<S>,
        HostPlacement: PlacementShape<S, Float32T, cs!(HostShape)>,
        HostPlacement: PlacementShape<S, Float64T, cs!(HostShape)>,
    {
        match x {
            AbstractTensor::Fixed64(x) => {
                unimplemented!()
                // plc.shape(sess, &x)
            }
            AbstractTensor::Fixed128(x) => {
                unimplemented!()
                // plc.shape(sess, &x)
            }
            AbstractTensor::Float32(x) => {
                plc.shape(sess, &x)
            }
            AbstractTensor::Float64(x) => {
                plc.shape(sess, &x)
            }
        }
    }
}

