use crate::computation::*;
use crate::error::Result;
use crate::host::{HostFloat32Tensor, HostFloat64Tensor, HostShape, HostString};
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

impl<HostT> PartiallySymbolicType for FloatTensor<HostT>
where
    HostT: SymbolicType,
    <HostT as SymbolicType>::Type: Placed<Placement = HostPlacement>,
{
    type Type = FloatTensor<<HostT as SymbolicType>::Type>;
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

modelled_kernel! {
    PlacementMean::mean, FloatingpointMean{axis: Option<u32>},
    [
        (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

impl FloatingpointMean {
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

modelled_kernel! {
    PlacementSum::sum, FloatingpointSum{axis: Option<u32>},
    [
        (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

impl FloatingpointSum {
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

modelled_kernel! {
    PlacementAtLeast2D::at_least_2d, FloatingpointAtLeast2D{to_column_vector: bool},
    [
        (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

impl FloatingpointAtLeast2D {
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

modelled_kernel! {
    PlacementAdd::add, FloatingpointAdd,
    [
        (HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

impl FloatingpointAdd {
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

modelled_kernel! {
    PlacementSub::sub, FloatingpointSub,
    [
        (HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

impl FloatingpointSub {
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

modelled_kernel! {
    PlacementMul::mul, FloatingpointMul,
    [
        (HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

impl FloatingpointMul {
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

modelled_kernel! {
    PlacementDiv::div, FloatingpointDiv,
    [
        (HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

impl FloatingpointDiv {
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

modelled_kernel! {
    PlacementDot::dot, FloatingpointDot,
    [
        (HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

impl FloatingpointDot {
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

modelled_kernel! {
    PlacementOnes::ones, FloatingpointOnes,
    [
        (HostPlacement, (HostShape) -> Float64Tensor => [hybrid] Self::float_host_kernel),
    ]
}

impl FloatingpointOnes {
    fn float_host_kernel<S: Session, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        shape: cs!(HostShape),
    ) -> Result<FloatTensor<HostFloatT>>
    where
        HostShape: KnownType<S>,
        HostFloat64Tensor: KnownType<S>,
        HostPlacement: PlacementOnes<S, cs!(HostShape), HostFloatT>,
    {
        let z = plc.ones(sess, &shape);
        Ok(FloatTensor::Host(z))
    }
}

modelled!(PlacementIndexAxis::index_axis, HostPlacement, attributes[axis: usize, index:usize] (Float32Tensor) -> Float32Tensor, IndexAxis);
modelled!(PlacementIndexAxis::index_axis, HostPlacement, attributes[axis: usize,  index:usize] (Float64Tensor) -> Float64Tensor, IndexAxis);

impl IndexAxis {
    pub(crate) fn float_host_kernel<S: Session, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        axis: usize,
        index: usize,
        x: FloatTensor<HostFloatT>,
    ) -> Result<FloatTensor<HostFloatT>>
    where
        HostPlacement: PlacementIndexAxis<S, HostFloatT, HostFloatT>,
    {
        let FloatTensor::Host(x) = x;
        let z = plc.index_axis(sess, axis, index, &x);
        Ok(FloatTensor::Host(z))
    }
}

modelled_kernel! {
    PlacementExpandDims::expand_dims, FloatingpointExpandDims{axis: Vec<u32>},
    [
        (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

impl FloatingpointExpandDims {
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

modelled!(PlacementConcatenate::concatenate, HostPlacement, attributes[axis: u32] vec[Float32Tensor] -> Float32Tensor, FloatingpointConcat);
modelled!(PlacementConcatenate::concatenate, HostPlacement, attributes[axis: u32] vec[Float64Tensor] -> Float64Tensor, FloatingpointConcat);

kernel! {
    FloatingpointConcat,
    [
        (HostPlacement, vec[Float32Tensor] -> Float32Tensor => [concrete] attributes[axis] Self::float_host_kernel),
        (HostPlacement, vec[Float64Tensor] -> Float64Tensor => [concrete] attributes[axis] Self::float_host_kernel),
    ]
}

impl FloatingpointConcat {
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

modelled_kernel! {
    PlacementTranspose::transpose, FloatingpointTranspose,
    [
        // (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::kernel),
    ]
}

impl FloatingpointTranspose {
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

modelled_kernel! {
    PlacementInverse::inverse, FloatingpointInverse,
    [
        // (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::kernel),
    ]
}

impl FloatingpointInverse {
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

impl Load {
    pub fn float_kernel<S: Session>(
        sess: &S,
        plc: &HostPlacement,
        key: cs!(HostString),
        query: cs!(HostString),
    ) -> Result<FloatTensor<cs!(HostFloat64Tensor)>>
    where
        HostString: KnownType<S>,
        HostFloat32Tensor: KnownType<S>,
        HostFloat64Tensor: KnownType<S>,
        HostPlacement: PlacementLoad<S, cs!(HostString), cs!(HostString), cs!(HostFloat64Tensor)>,
    {
        let z = plc.load(sess, &key, &query);
        Ok(FloatTensor::Host(z))
    }
}

impl Save {
    pub fn float_kernel<S: Session, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        key: cs!(HostString),
        x: FloatTensor<HostFloatT>,
    ) -> Result<cs!(Unit)>
    where
        HostString: KnownType<S>,
        Unit: KnownType<S>,
        HostPlacement: PlacementSave<S, cs!(HostString), HostFloatT, cs!(Unit)>,
    {
        let FloatTensor::Host(x) = x;
        Ok(plc.save(sess, &key, &x))
    }
}

impl Shape {
    pub(crate) fn float_kernel<S: Session, HostFloatT, HostShapeT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT>,
    ) -> Result<HostShapeT>
    where
        HostPlacement: PlacementShape<S, HostFloatT, HostShapeT>,
    {
        let FloatTensor::Host(x) = x;
        Ok(plc.shape(sess, &x))
    }
}

impl ConstantOp {
    pub fn float_kernel<S: Session, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        value: Constant,
    ) -> Result<FloatTensor<HostFloatT>>
    where
        HostPlacement: PlacementConstant<S, HostFloatT>,
    {
        let z = plc.constant(sess, value);
        Ok(FloatTensor::Host(z))
    }
}

impl Input {
    pub fn float_kernel<S: Session, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        arg_name: String,
    ) -> Result<FloatTensor<HostFloatT>>
    where
        HostPlacement: PlacementInput<S, HostFloatT>,
    {
        let z = plc.input(sess, arg_name);
        Ok(FloatTensor::Host(z))
    }
}

impl Output {
    pub fn float_kernel<S: Session, HostFloatT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT>,
    ) -> Result<FloatTensor<HostFloatT>>
    where
        HostPlacement: PlacementOutput<S, HostFloatT, HostFloatT>,
    {
        let FloatTensor::Host(x) = x;
        Ok(FloatTensor::Host(plc.output(sess, &x)))
    }
}
