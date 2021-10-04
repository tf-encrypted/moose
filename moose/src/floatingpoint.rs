use crate::computation::*;
use crate::error::Result;
use crate::host::{HostFloat32Tensor, HostFloat64Tensor, HostShape};
use crate::kernels::*;
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

modelled!(PlacementMean::mean, HostPlacement, attributes[axis: Option<u32>] (Float32Tensor) -> Float32Tensor, FloatingpointMeanOp);
modelled!(PlacementMean::mean, HostPlacement, attributes[axis: Option<u32>] (Float64Tensor) -> Float64Tensor, FloatingpointMeanOp);

kernel! {
    FloatingpointMeanOp,
    [
        (HostPlacement, (Float32Tensor) -> Float32Tensor => [hybrid] attributes[axis] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [hybrid] attributes[axis] Self::float_host_kernel),
    ]
}

impl FloatingpointMeanOp {
    fn float_host_kernel<S: Session, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: FloatTensor<HostFloatT>,
    ) -> Result<FloatTensor<HostFloatT>>
    where
        HostPlacement: PlacementMean<S, HostFloatT, HostFloatT>,
    {
        let FloatTensor::Host(x) = x;

        let z = plc.mean(sess, axis, &x);
        Ok(FloatTensor::Host(z))
    }
}

modelled!(PlacementSum::sum, HostPlacement, attributes[axis: Option<u32>] (Float32Tensor) -> Float32Tensor, FloatingpointSumOp);
modelled!(PlacementSum::sum, HostPlacement, attributes[axis: Option<u32>] (Float64Tensor) -> Float64Tensor, FloatingpointSumOp);

kernel! {
    FloatingpointSumOp,
    [
        (HostPlacement, (Float32Tensor) -> Float32Tensor => [hybrid] attributes[axis] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [hybrid] attributes[axis] Self::float_host_kernel),
    ]
}

impl FloatingpointSumOp {
    fn float_host_kernel<S: Session, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: FloatTensor<HostFloatT>,
    ) -> Result<FloatTensor<HostFloatT>>
    where
        HostPlacement: PlacementSum<S, HostFloatT, HostFloatT>,
    {
        let FloatTensor::Host(x) = x;

        let z = plc.sum(sess, axis, &x);
        Ok(FloatTensor::Host(z))
    }
}

modelled!(PlacementAtLeast2D::at_least_2d, HostPlacement, attributes[to_column_vector: bool] (Float32Tensor) -> Float32Tensor, FloatingpointAtLeast2DOp);
modelled!(PlacementAtLeast2D::at_least_2d, HostPlacement, attributes[to_column_vector: bool] (Float64Tensor) -> Float64Tensor, FloatingpointAtLeast2DOp);

kernel! {
    FloatingpointAtLeast2DOp,
    [
        (HostPlacement, (Float32Tensor) -> Float32Tensor => [hybrid] attributes[to_column_vector] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [hybrid] attributes[to_column_vector] Self::float_host_kernel),
    ]
}

impl FloatingpointAtLeast2DOp {
    fn float_host_kernel<S: Session, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        to_column_vector: bool,
        x: FloatTensor<HostFloatT>,
    ) -> Result<FloatTensor<HostFloatT>>
    where
        HostPlacement: PlacementAtLeast2D<S, HostFloatT, HostFloatT>,
    {
        let FloatTensor::Host(x) = x;

        let z = plc.at_least_2d(sess, to_column_vector, &x);
        Ok(FloatTensor::Host(z))
    }
}

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
    ) -> Result<FloatTensor<HostFloatT>>
    where
        HostPlacement: PlacementAdd<S, HostFloatT, HostFloatT, HostFloatT>,
    {
        let FloatTensor::Host(x) = x;
        let FloatTensor::Host(y) = y;

        let z = plc.add(sess, &x, &y);
        Ok(FloatTensor::Host(z))
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
    ) -> Result<FloatTensor<HostFloatT>>
    where
        HostPlacement: PlacementSub<S, HostFloatT, HostFloatT, HostFloatT>,
    {
        let FloatTensor::Host(x) = x;
        let FloatTensor::Host(y) = y;

        let z = plc.sub(sess, &x, &y);
        Ok(FloatTensor::Host(z))
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
    ) -> Result<FloatTensor<HostFloatT>>
    where
        HostPlacement: PlacementMul<S, HostFloatT, HostFloatT, HostFloatT>,
    {
        let FloatTensor::Host(x) = x;
        let FloatTensor::Host(y) = y;

        let z = plc.mul(sess, &x, &y);
        Ok(FloatTensor::Host(z))
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
    ) -> Result<FloatTensor<HostFloatT>>
    where
        HostPlacement: PlacementDiv<S, HostFloatT, HostFloatT, HostFloatT>,
    {
        let FloatTensor::Host(x) = x;
        let FloatTensor::Host(y) = y;

        let z = plc.div(sess, &x, &y);
        Ok(FloatTensor::Host(z))
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
    ) -> Result<FloatTensor<HostFloatT>>
    where
        HostPlacement: PlacementDot<S, HostFloatT, HostFloatT, HostFloatT>,
    {
        let FloatTensor::Host(x) = x;
        let FloatTensor::Host(y) = y;

        let z = plc.dot(sess, &x, &y);
        Ok(FloatTensor::Host(z))
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
    ) -> Result<FloatTensor<cs!(HostFloat64Tensor)>>
    where
        HostShape: KnownType<S>,
        HostFloat64Tensor: KnownType<S>,
        HostPlacement: PlacementOnes<S, cs!(HostShape), cs!(HostFloat64Tensor)>,
    {
        let z = plc.ones(sess, &shape);
        Ok(FloatTensor::Host(z))
    }
}

modelled!(PlacementExpandDims::expand_dims, HostPlacement, attributes[axis: Vec<u32>] (Float32Tensor) -> Float32Tensor, FloatingpointExpandDimsOp);
modelled!(PlacementExpandDims::expand_dims, HostPlacement, attributes[axis: Vec<u32>] (Float64Tensor) -> Float64Tensor, FloatingpointExpandDimsOp);

kernel! {
    FloatingpointExpandDimsOp,
    [
        (HostPlacement, (Float32Tensor) -> Float32Tensor => [hybrid] attributes[axis] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [hybrid] attributes[axis] Self::float_host_kernel),
    ]
}

impl FloatingpointExpandDimsOp {
    fn float_host_kernel<S: Session, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Vec<u32>,
        x: FloatTensor<HostFloatT>,
    ) -> Result<FloatTensor<HostFloatT>>
    where
        HostPlacement: PlacementExpandDims<S, HostFloatT, HostFloatT>,
    {
        let FloatTensor::Host(x) = x;
        let z = plc.expand_dims(sess, axis, &x);
        Ok(FloatTensor::Host(z))
    }
}

modelled!(PlacementConcatenate::concatenate, HostPlacement, attributes[axis: u32] vec[Float32Tensor] -> Float32Tensor, FloatingpointConcatOp);
modelled!(PlacementConcatenate::concatenate, HostPlacement, attributes[axis: u32] vec[Float64Tensor] -> Float64Tensor, FloatingpointConcatOp);

kernel! {
    FloatingpointConcatOp,
    [
        (HostPlacement, vec[Float32Tensor] -> Float32Tensor => [hybrid] attributes[axis] Self::float_host_kernel),
        (HostPlacement, vec[Float64Tensor] -> Float64Tensor => [hybrid] attributes[axis] Self::float_host_kernel),
    ]
}

impl FloatingpointConcatOp {
    fn float_host_kernel<S: Session, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        axis: u32,
        xs: &[FloatTensor<HostFloatT>],
    ) -> Result<FloatTensor<HostFloatT>>
    where
        HostPlacement: PlacementConcatenate<S, HostFloatT, HostFloatT>,
        HostFloatT: Clone,
    {
        let xs: Vec<HostFloatT> = xs
            .iter()
            .map(|x| match x {
                FloatTensor::Host(x) => (*x).clone(),
            })
            .collect();

        let z = plc.concatenate(sess, axis, &xs);
        Ok(FloatTensor::Host(z))
    }
}

// modelled!(PlacementTranspose::transpose, HostPlacement, (Float32Tensor) -> Float32Tensor, FloatingpointTransposeOp);
modelled!(PlacementTranspose::transpose, HostPlacement, (Float64Tensor) -> Float64Tensor, FloatingpointTransposeOp);

kernel! {
    FloatingpointTransposeOp, [
        // (HostPlacement, (Float32Tensor) -> Float32Tensor => [hybrid] Self::kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [hybrid] Self::kernel),
    ]
}

impl FloatingpointTransposeOp {
    pub fn kernel<S: Session, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT>,
    ) -> Result<FloatTensor<HostFloatT>>
    where
        HostPlacement: PlacementTranspose<S, HostFloatT, HostFloatT>,
    {
        let FloatTensor::Host(x) = x;
        let z = plc.transpose(sess, &x);
        Ok(FloatTensor::Host(z))
    }
}

// modelled!(PlacementInverse::inverse, HostPlacement, (Float32Tensor) -> Float32Tensor, FloatingpointInverseOp);
modelled!(PlacementInverse::inverse, HostPlacement, (Float64Tensor) -> Float64Tensor, FloatingpointInverseOp);

kernel! {
    FloatingpointInverseOp, [
        // (HostPlacement, (Float32Tensor) -> Float32Tensor => [hybrid] Self::kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [hybrid] Self::kernel),
    ]
}

impl FloatingpointInverseOp {
    pub fn kernel<S: Session, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT>,
    ) -> Result<FloatTensor<HostFloatT>>
    where
        HostPlacement: PlacementInverse<S, HostFloatT, HostFloatT>,
    {
        let FloatTensor::Host(x) = x;
        let z = plc.inverse(sess, &x);
        Ok(FloatTensor::Host(z))
    }
}

impl LoadOp {
    pub fn float_kernel<S: Session>(
        sess: &S,
        plc: &HostPlacement,
        key: cs!(String),
        query: cs!(String),
    ) -> Result<FloatTensor<cs!(HostFloat64Tensor)>>
    where
        String: KnownType<S>,
        HostFloat32Tensor: KnownType<S>,
        HostFloat64Tensor: KnownType<S>,
        HostPlacement: PlacementLoad<S, cs!(String), cs!(String), cs!(HostFloat64Tensor)>,
    {
        let z = plc.load(sess, &key, &query);
        Ok(FloatTensor::Host(z))
    }
}

impl SaveOp {
    pub fn float_kernel<S: Session, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        key: cs!(String),
        x: FloatTensor<HostFloatT>,
    ) -> Result<cs!(Unit)>
    where
        String: KnownType<S>,
        Unit: KnownType<S>,
        HostPlacement: PlacementSave<S, cs!(String), HostFloatT, cs!(Unit)>,
    {
        let FloatTensor::Host(x) = x;
        Ok(plc.save(sess, &key, &x))
    }
}

impl ShapeOp {
    pub(crate) fn float_kernel<S: Session, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT>,
    ) -> Result<cs!(HostShape)>
    where
        HostShape: KnownType<S>,
        HostPlacement: PlacementShape<S, HostFloatT, cs!(HostShape)>,
    {
        let FloatTensor::Host(x) = x;
        Ok(plc.shape(sess, &x))
    }
}

impl ConstantOp {
    pub fn float_kernel<S: Session>(
        sess: &S,
        plc: &HostPlacement,
        value: Constant,
    ) -> Result<FloatTensor<cs!(HostFloat64Tensor)>>
    where
        HostFloat32Tensor: KnownType<S>,
        HostFloat64Tensor: KnownType<S>,
        HostPlacement: PlacementConstant<S, cs!(HostFloat64Tensor)>,
    {
        let z = plc.constant(sess, value);
        Ok(FloatTensor::Host(z))
    }
}

impl OutputOp {
    pub fn float_kernel<S: Session, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT>,
    ) -> Result<cs!(Unit)>
    where
        String: KnownType<S>,
        Unit: KnownType<S>,
        HostPlacement: PlacementOutput<S, HostFloatT, cs!(Unit)>,
    {
        let FloatTensor::Host(x) = x;
        Ok(plc.output(sess, &x))
    }
}
