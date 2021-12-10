use crate::boolean::{BoolTensor, BooleanTensor};
use crate::computation::*;
use crate::error::Result;
use crate::host::{HostFloat32Tensor, HostFloat64Tensor, HostShape, HostString};
use crate::kernels::*;
use crate::mirrored::Mirrored3Tensor;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FloatTensor<HostT, MirroredT> {
    Host(HostT),
    Mirrored3(MirroredT),
}

moose_type!(Mirrored3Float32 = Mirrored3Tensor<HostFloat32Tensor>);
moose_type!(Mirrored3Float64 = Mirrored3Tensor<HostFloat64Tensor>);

moose_type!(Float32Tensor = FloatTensor<HostFloat32Tensor, Mirrored3Float32>);
moose_type!(Float64Tensor = FloatTensor<HostFloat64Tensor, Mirrored3Float64>);

impl<T, MirroredT> Placed for FloatTensor<T, MirroredT>
where
    T: Placed,
    T::Placement: Into<Placement>,
    MirroredT: Placed,
    MirroredT::Placement: Into<Placement>,
{
    type Placement = Placement;

    fn placement(&self) -> Result<Self::Placement> {
        match self {
            FloatTensor::Host(x) => Ok(x.placement()?.into()),
            FloatTensor::Mirrored3(x) => Ok(x.placement()?.into()),
        }
    }
}

modelled_kernel! {
    PlacementMean::mean, FloatingpointMeanOp{axis: Option<u32>},
    [
        (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

impl FloatingpointMeanOp {
    fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementMean<S, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(v) => unimplemented!(),
        };
        let z = plc.mean(sess, axis, &x);
        Ok(FloatTensor::Host(z))
    }
}

modelled_kernel! {
    PlacementSum::sum, FloatingpointSumOp{axis: Option<u32>},
    [
        (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

impl FloatingpointSumOp {
    fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<u32>,
        x: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementSum<S, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(v) => unimplemented!(),
        };
        let z = plc.sum(sess, axis, &x);
        Ok(FloatTensor::Host(z))
    }
}

modelled_kernel! {
    PlacementAtLeast2D::at_least_2d, FloatingpointAtLeast2DOp{to_column_vector: bool},
    [
        (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

impl FloatingpointAtLeast2DOp {
    fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        to_column_vector: bool,
        x: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementAtLeast2D<S, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(v) => unimplemented!(),
        };

        let z = plc.at_least_2d(sess, to_column_vector, &x);
        Ok(FloatTensor::Host(z))
    }
}

modelled_kernel! {
    PlacementAdd::add, FloatingpointAddOp,
    [
        (HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

impl FloatingpointAddOp {
    fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT, MirroredT>,
        y: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementAdd<S, HostFloatT, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(v) => unimplemented!(),
        };

        let y = match y {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(v) => unimplemented!(),
        };

        let z = plc.add(sess, &x, &y);
        Ok(FloatTensor::Host(z))
    }
}

modelled_kernel! {
    PlacementSub::sub, FloatingpointSubOp,
    [
        (HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

impl FloatingpointSubOp {
    fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT, MirroredT>,
        y: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementSub<S, HostFloatT, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(v) => unimplemented!(),
        };
        let y = match y {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(v) => unimplemented!(),
        };

        let z = plc.sub(sess, &x, &y);
        Ok(FloatTensor::Host(z))
    }
}

modelled_kernel! {
    PlacementMul::mul, FloatingpointMulOp,
    [
        (HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

impl FloatingpointMulOp {
    fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT, MirroredT>,
        y: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementMul<S, HostFloatT, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(v) => unimplemented!(),
        };
        let y = match y {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(v) => unimplemented!(),
        };

        let z = plc.mul(sess, &x, &y);
        Ok(FloatTensor::Host(z))
    }
}

modelled_kernel! {
    PlacementDiv::div, FloatingpointDivOp,
    [
        (HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

impl FloatingpointDivOp {
    fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT, MirroredT>,
        y: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementDiv<S, HostFloatT, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(v) => unimplemented!(),
        };
        let y = match y {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(v) => unimplemented!(),
        };

        let z = plc.div(sess, &x, &y);
        Ok(FloatTensor::Host(z))
    }
}

modelled_kernel! {
    PlacementDot::dot, FloatingpointDotOp,
    [
        (HostPlacement, (Float32Tensor, Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor, Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

impl FloatingpointDotOp {
    fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT, MirroredT>,
        y: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementDot<S, HostFloatT, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(v) => unimplemented!(),
        };
        let y = match y {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(v) => unimplemented!(),
        };

        let z = plc.dot(sess, &x, &y);
        Ok(FloatTensor::Host(z))
    }
}

modelled!(PlacementLessThan::less, HostPlacement, (Float32Tensor, Float32Tensor) -> BooleanTensor, LessOp);
modelled!(PlacementLessThan::less, HostPlacement, (Float64Tensor, Float64Tensor) -> BooleanTensor, LessOp);

impl LessOp {
    pub(crate) fn float_kernel<S: Session, HostFloatT, HostBitT, RepBitT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT, MirroredT>,
        y: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<BoolTensor<HostBitT, RepBitT>>
    where
        HostPlacement: PlacementLessThan<S, HostFloatT, HostFloatT, HostBitT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(v) => unimplemented!(),
        };
        let y = match y {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(v) => unimplemented!(),
        };

        let z = plc.less(sess, &x, &y);
        Ok(BoolTensor::Host(z))
    }
}

modelled_kernel! {
    PlacementOnes::ones, FloatingpointOnesOp,
    [
        (HostPlacement, (HostShape) -> Float64Tensor => [hybrid] Self::float_host_kernel),
    ]
}

impl FloatingpointOnesOp {
    fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        shape: cs!(HostShape),
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostShape: KnownType<S>,
        HostFloat64Tensor: KnownType<S>,
        HostPlacement: PlacementOnes<S, cs!(HostShape), HostFloatT>,
    {
        let z = plc.ones(sess, &shape);
        Ok(FloatTensor::Host(z))
    }
}

impl IndexAxisOp {
    pub(crate) fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        axis: usize,
        index: usize,
        x: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementIndexAxis<S, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(v) => unimplemented!(),
        };

        let z = plc.index_axis(sess, axis, index, &x);
        Ok(FloatTensor::Host(z))
    }
}

modelled_kernel! {
    PlacementExpandDims::expand_dims, FloatingpointExpandDimsOp{axis: Vec<u32>},
    [
        (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::float_host_kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::float_host_kernel),
    ]
}

impl FloatingpointExpandDimsOp {
    fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Vec<u32>,
        x: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementExpandDims<S, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(v) => unimplemented!(),
        };

        let z = plc.expand_dims(sess, axis, &x);
        Ok(FloatTensor::Host(z))
    }
}

modelled!(PlacementConcatenate::concatenate, HostPlacement, attributes[axis: u32] vec[Float32Tensor] -> Float32Tensor, FloatingpointConcatOp);
modelled!(PlacementConcatenate::concatenate, HostPlacement, attributes[axis: u32] vec[Float64Tensor] -> Float64Tensor, FloatingpointConcatOp);

kernel! {
    FloatingpointConcatOp,
    [
        (HostPlacement, vec[Float32Tensor] -> Float32Tensor => [concrete] attributes[axis] Self::float_host_kernel),
        (HostPlacement, vec[Float64Tensor] -> Float64Tensor => [concrete] attributes[axis] Self::float_host_kernel),
    ]
}

impl FloatingpointConcatOp {
    fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        axis: u32,
        xs: &[FloatTensor<HostFloatT, MirroredT>],
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementConcatenate<S, HostFloatT, HostFloatT>,
        HostFloatT: Clone,
    {
        let xs: Vec<HostFloatT> = xs
            .iter()
            .map(|x| match x {
                FloatTensor::Host(x) => (*x).clone(),
                FloatTensor::Mirrored3(x) => unimplemented!(), // TODO(Dragos) fix this
            })
            .collect();

        let z = plc.concatenate(sess, axis, &xs);
        Ok(FloatTensor::Host(z))
    }
}

modelled_kernel! {
    PlacementTranspose::transpose, FloatingpointTransposeOp,
    [
        // (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::kernel),
    ]
}

impl FloatingpointTransposeOp {
    pub fn kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementTranspose<S, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(v) => unimplemented!(),
        };

        let z = plc.transpose(sess, &x);
        Ok(FloatTensor::Host(z))
    }
}

modelled_kernel! {
    PlacementInverse::inverse, FloatingpointInverseOp,
    [
        // (HostPlacement, (Float32Tensor) -> Float32Tensor => [concrete] Self::kernel),
        (HostPlacement, (Float64Tensor) -> Float64Tensor => [concrete] Self::kernel),
    ]
}

impl FloatingpointInverseOp {
    pub fn kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementInverse<S, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(v) => unimplemented!(),
        };

        let z = plc.inverse(sess, &x);
        Ok(FloatTensor::Host(z))
    }
}

impl LoadOp {
    pub fn float_kernel<S: Session, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        key: cs!(HostString),
        query: cs!(HostString),
    ) -> Result<FloatTensor<cs!(HostFloat64Tensor), MirroredT>>
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

impl SaveOp {
    pub fn float_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        key: cs!(HostString),
        x: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<cs!(Unit)>
    where
        HostString: KnownType<S>,
        Unit: KnownType<S>,
        HostPlacement: PlacementSave<S, cs!(HostString), HostFloatT, cs!(Unit)>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(v) => unimplemented!(),
        };

        Ok(plc.save(sess, &key, &x))
    }
}

impl ShapeOp {
    pub(crate) fn float_kernel<S: Session, HostFloatT, HostShapeT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<HostShapeT>
    where
        HostPlacement: PlacementShape<S, HostFloatT, HostShapeT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(v) => unimplemented!(),
        };

        Ok(plc.shape(sess, &x))
    }
}

impl ConstantOp {
    pub fn float_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        value: Constant,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementConstant<S, HostFloatT>,
    {
        let z = plc.constant(sess, value);
        Ok(FloatTensor::Host(z))
    }
}

impl ConstantOp {
    pub fn mir3_float_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &Mirrored3Placement,
        value: Constant,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementConstant<S, HostFloatT>,
        Mirrored3Tensor<HostFloatT>: Into<MirroredT>,
    {
        let (player0, player1, player2) = plc.host_placements();

        let z0 = player0.constant(sess, value.clone());
        let z1 = player1.constant(sess, value.clone());
        let z2 = player2.constant(sess, value);

        Ok(FloatTensor::Mirrored3(
            Mirrored3Tensor {
                values: [z0, z1, z2],
            }.into()
        ))
    }
}

impl InputOp {
    pub fn float_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        arg_name: String,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementInput<S, HostFloatT>,
    {
        let z = plc.input(sess, arg_name);
        Ok(FloatTensor::Host(z))
    }
}

impl OutputOp {
    pub fn float_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementOutput<S, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(v) => unimplemented!(),
        };

        Ok(FloatTensor::Host(plc.output(sess, &x)))
    }
}
