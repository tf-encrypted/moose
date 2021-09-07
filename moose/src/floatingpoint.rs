use crate::computation::{FloatingpointAddOp, FloatingpointDivOp, FloatingpointDotOp, FloatingpointMulOp, FloatingpointOnesOp, FloatingpointSubOp, HostPlacement, KnownType, Placed, Placement, SymbolicType};
use crate::error::Result;
use crate::host::{HostFloat32Tensor, HostFloat64Tensor, HostShape};
use crate::kernels::{PlacementAdd, PlacementDiv, PlacementDot, PlacementMul, PlacementOnes, PlacementSub, Session};
use crate::symbolic::Symbolic;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FloatTensor<HostT> {
    Host(HostT),
}

pub type Float32Tensor = FloatTensor<HostFloat32Tensor>;

pub type Float64Tensor = FloatTensor<HostFloat64Tensor>;

impl<T> Placed for FloatTensor<T>
where
    T: Placed<Placement = HostPlacement>,
{
    type Placement = Placement;

    fn placement(&self) -> Result<Self::Placement> {
        match self {
            FloatTensor::Host(x) => Ok(Placement::Host(x.placement()?)),
        }
    }
}

impl<HostT> SymbolicType for FloatTensor<HostT>
where
    HostT: SymbolicType,
    <HostT as SymbolicType>::Type: Placed<Placement = HostPlacement>,
{
    type Type = Symbolic<FloatTensor<<HostT as SymbolicType>::Type>>;
}

// TODO(lvorona): Not sure why we need this one separately... But the moose_type macro is coming!
impl<HostT: Placed<Placement = HostPlacement>> From<FloatTensor<HostT>>
    for Symbolic<FloatTensor<HostT>>
{
    fn from(x: FloatTensor<HostT>) -> Self {
        Symbolic::Concrete(x)
    }
}

impl<HostFloatT> TryFrom<Symbolic<FloatTensor<HostFloatT>>> for FloatTensor<HostFloatT>
where
    HostFloatT: Placed<Placement = HostPlacement>,
{
    type Error = ();
    fn try_from(v: Symbolic<FloatTensor<HostFloatT>>) -> std::result::Result<Self, ()> {
        match v {
            Symbolic::Concrete(x) => Ok(x),
            _ => Err(()),
        }
    }
}

// TODO(Morten)

// modelled!(PlacementMean::mean, HostPlacement, attributes[axis: Option<u32>] (Float32Tensor) -> Float32Tensor, FloatingpointMeanOp);
// modelled!(PlacementMean::mean, HostPlacement, attributes[axis: Option<u32>] (Float64Tensor) -> Float64Tensor, FloatingpointMeanOp);

// modelled!(PlacementAtLeast2D::at_least_2d, HostPlacement, attributes[to_column_vector: bool] (Float32Tensor) -> Float32Tensor, HostAtLeast2DOp);
// modelled!(PlacementAtLeast2D::at_least_2d, HostPlacement, attributes[to_column_vector: bool] (Float64Tensor) -> Float64Tensor, HostAtLeast2DOp);

modelled!(PlacementAdd::add, HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor, FloatingpointAddOp);
modelled!(PlacementAdd::add, HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor, FloatingpointAddOp);

kernel! {
    FloatingpointAddOp,
    [
        (HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor => [hybrid] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor => [hybrid] Self::float_host_kernel),
    ]
}

impl FloatingpointAddOp {
    fn float_host_kernel<S: Session, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT>,
        y: FloatTensor<HostFloatT>,
    ) -> FloatTensor<HostFloatT>
    where
        HostPlacement: PlacementAdd<S, HostFloatT, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
        };
        let y = match y {
            FloatTensor::Host(v) => v,
        };

        let z = plc.add(sess, &x, &y);
        FloatTensor::Host(z)
    }
}

modelled!(PlacementSub::sub, HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor, FloatingpointSubOp);
modelled!(PlacementSub::sub, HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor, FloatingpointSubOp);

kernel! {
    FloatingpointSubOp,
    [
        (HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor => [hybrid] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor => [hybrid] Self::float_host_kernel),
    ]
}

impl FloatingpointSubOp {
    fn float_host_kernel<S: Session, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT>,
        y: FloatTensor<HostFloatT>,
    ) -> FloatTensor<HostFloatT>
    where
        HostPlacement: PlacementSub<S, HostFloatT, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
        };
        let y = match y {
            FloatTensor::Host(v) => v,
        };

        let z = plc.sub(sess, &x, &y);
        FloatTensor::Host(z)
    }
}

modelled!(PlacementMul::mul, HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor, FloatingpointMulOp);
modelled!(PlacementMul::mul, HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor, FloatingpointMulOp);

kernel! {
    FloatingpointMulOp,
    [
        (HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor => [hybrid] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor => [hybrid] Self::float_host_kernel),
    ]
}

impl FloatingpointMulOp {
    fn float_host_kernel<S: Session, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT>,
        y: FloatTensor<HostFloatT>,
    ) -> FloatTensor<HostFloatT>
    where
        HostPlacement: PlacementMul<S, HostFloatT, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
        };
        let y = match y {
            FloatTensor::Host(v) => v,
        };

        let z = plc.mul(sess, &x, &y);
        FloatTensor::Host(z)
    }
}

modelled!(PlacementDiv::div, HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor, FloatingpointDivOp);
modelled!(PlacementDiv::div, HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor, FloatingpointDivOp);

kernel! {
    FloatingpointDivOp,
    [
        (HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor => [hybrid] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor => [hybrid] Self::float_host_kernel),
    ]
}

impl FloatingpointDivOp {
    fn float_host_kernel<S: Session, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT>,
        y: FloatTensor<HostFloatT>,
    ) -> FloatTensor<HostFloatT>
    where
        HostPlacement: PlacementDiv<S, HostFloatT, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
        };
        let y = match y {
            FloatTensor::Host(v) => v,
        };

        let z = plc.div(sess, &x, &y);
        FloatTensor::Host(z)
    }
}

modelled!(PlacementDot::dot, HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor, FloatingpointDotOp);
modelled!(PlacementDot::dot, HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor, FloatingpointDotOp);

kernel! {
    FloatingpointDotOp,
    [
        (HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor => [hybrid] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor => [hybrid] Self::float_host_kernel),
    ]
}

impl FloatingpointDotOp {
    fn float_host_kernel<S: Session, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT>,
        y: FloatTensor<HostFloatT>,
    ) -> FloatTensor<HostFloatT>
    where
        HostPlacement: PlacementDot<S, HostFloatT, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
        };
        let y = match y {
            FloatTensor::Host(v) => v,
        };

        let z = plc.dot(sess, &x, &y);
        FloatTensor::Host(z)
    }
}

modelled!(PlacementOnes::ones, HostPlacement, (HostShape) -> Float64Tensor, FloatingpointOnesOp);

kernel! {
    FloatingpointOnesOp,
    [
        (HostPlacement, (HostShape) -> Float64Tensor => [hybrid] Self::float_host_kernel),
    ]
}

impl FloatingpointOnesOp {
    fn float_host_kernel<S: Session>(
        sess: &S,
        plc: &HostPlacement,
        shape: cs!(HostShape),
    ) -> FloatTensor<cs!(HostFloat64Tensor)>
    where
        HostShape: KnownType<S>,
        HostFloat64Tensor: KnownType<S>,
        HostPlacement: PlacementOnes<S, cs!(HostShape), cs!(HostFloat64Tensor)>,
    {
        let z = plc.ones(sess, &shape);
        FloatTensor::Host(z)
    }
}