use super::*;
use crate::boolean::BoolTensor;
use crate::computation::*;
use crate::error::Error;
use crate::error::Result;
use crate::execution::Session;
use crate::host::HostPlacement;
use crate::kernels::*;
use crate::mirrored::{Mir3Tensor, Mirrored3Placement};
use crate::types::*;

impl IdentityOp {
    pub(crate) fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementPlace<S, HostFloatT>,
        HostPlacement: PlacementDemirror<S, MirroredT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => plc.place(sess, v),
            FloatTensor::Mirrored3(v) => plc.demirror(sess, &v),
        };
        Ok(FloatTensor::Host(x))
    }
}

impl MeanOp {
    pub(crate) fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
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
            FloatTensor::Mirrored3(_v) => unimplemented!(),
        };
        let z = plc.mean(sess, axis, &x);
        Ok(FloatTensor::Host(z))
    }
}

impl SumOp {
    pub(crate) fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Option<usize>,
        x: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementSum<S, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(_v) => unimplemented!(),
        };
        let z = plc.sum(sess, axis, &x);
        Ok(FloatTensor::Host(z))
    }
}

impl AtLeast2DOp {
    pub(crate) fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
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
            FloatTensor::Mirrored3(_v) => unimplemented!(),
        };

        let z = plc.at_least_2d(sess, to_column_vector, &x);
        Ok(FloatTensor::Host(z))
    }
}

impl AbsOp {
    pub(crate) fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementAbs<S, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(_v) => unimplemented!(),
        };
        Ok(FloatTensor::Host(plc.abs(sess, &x)))
    }
}

impl AddOp {
    pub(crate) fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
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
            FloatTensor::Mirrored3(_v) => unimplemented!(),
        };

        let y = match y {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(_v) => unimplemented!(),
        };

        let z = plc.add(sess, &x, &y);
        Ok(FloatTensor::Host(z))
    }
}

impl AddNOp {
    pub(crate) fn float_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        xs: &[FloatTensor<HostFloatT, MirroredT>],
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementAddN<S, HostFloatT, HostFloatT>,
        HostFloatT: Clone,
    {
        if xs.is_empty() {
            Err(Error::InvalidArgument(
                "cannot add_n on empty array of tensors".to_string(),
            ))
        } else {
            let first = &xs[0];
            let vec: Vec<HostFloatT> = match first {
                FloatTensor::Host(_) => {
                    let vec: Vec<HostFloatT> = xs
                        .iter()
                        .map(|tensor| match tensor {
                            FloatTensor::Host(x) => x.clone(),
                            FloatTensor::Mirrored3(_) => unimplemented!(),
                        })
                        .collect();
                    vec
                }
                FloatTensor::Mirrored3(_v) => unimplemented!(),
            };
            let result = plc.add_n(sess, &vec);
            Ok(FloatTensor::Host(result))
        }
    }
}

impl SubOp {
    pub(crate) fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
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
            FloatTensor::Mirrored3(_v) => unimplemented!(),
        };
        let y = match y {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(_v) => unimplemented!(),
        };

        let z = plc.sub(sess, &x, &y);
        Ok(FloatTensor::Host(z))
    }
}

impl MulOp {
    pub(crate) fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
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
            FloatTensor::Mirrored3(_v) => unimplemented!(),
        };
        let y = match y {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(_v) => unimplemented!(),
        };

        let z = plc.mul(sess, &x, &y);
        Ok(FloatTensor::Host(z))
    }
}

impl DivOp {
    pub(crate) fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
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
            FloatTensor::Mirrored3(_v) => unimplemented!(),
        };
        let y = match y {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(_v) => unimplemented!(),
        };

        let z = plc.div(sess, &x, &y);
        Ok(FloatTensor::Host(z))
    }
}

impl DotOp {
    pub(crate) fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
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
            FloatTensor::Mirrored3(_v) => unimplemented!(),
        };
        let y = match y {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(_v) => unimplemented!(),
        };

        let z = plc.dot(sess, &x, &y);
        Ok(FloatTensor::Host(z))
    }
}

impl LessThanOp {
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
            FloatTensor::Mirrored3(_v) => unimplemented!(),
        };
        let y = match y {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(_v) => unimplemented!(),
        };

        let z = plc.less(sess, &x, &y);
        Ok(BoolTensor::Host(z))
    }
}

impl OnesOp {
    pub(crate) fn host_float_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        shape: m!(HostShape),
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostShape: KnownType<S>,
        HostFloat64Tensor: KnownType<S>,
        HostPlacement: PlacementOnes<S, m!(HostShape), HostFloatT>,
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
            FloatTensor::Mirrored3(_v) => unimplemented!(),
        };

        let z = plc.index_axis(sess, axis, index, &x);
        Ok(FloatTensor::Host(z))
    }
}

impl ExpandDimsOp {
    pub(crate) fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        axis: Vec<usize>,
        x: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementExpandDims<S, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(_v) => unimplemented!(),
        };

        let z = plc.expand_dims(sess, axis, &x);
        Ok(FloatTensor::Host(z))
    }
}

impl ConcatOp {
    pub(crate) fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
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
                FloatTensor::Mirrored3(_x) => unimplemented!(), // TODO(Dragos) fix this
            })
            .collect();

        let z = plc.concatenate(sess, axis, &xs);
        Ok(FloatTensor::Host(z))
    }
}

impl TransposeOp {
    pub(crate) fn float_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementTranspose<S, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(_v) => unimplemented!(),
        };

        let z = plc.transpose(sess, &x);
        Ok(FloatTensor::Host(z))
    }
}

impl InverseOp {
    pub(crate) fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementInverse<S, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(_v) => unimplemented!(),
        };

        let z = plc.inverse(sess, &x);
        Ok(FloatTensor::Host(z))
    }
}

impl LogOp {
    pub(crate) fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementLog<S, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(_v) => {
                return Err(Error::UnimplementedOperator(
                    "LogOp @ Mirrored3Placement".to_string(),
                ))
            }
        };
        let z = plc.log(sess, &x);
        Ok(FloatTensor::Host(z))
    }
}

impl Log2Op {
    pub(crate) fn float_host_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementLog2<S, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(_v) => {
                return Err(Error::UnimplementedOperator(
                    "Log2Op @ Mirrored3Placement".to_string(),
                ))
            }
        };
        let z = plc.log2(sess, &x);
        Ok(FloatTensor::Host(z))
    }
}

impl LoadOp {
    pub(crate) fn float_kernel<S: Session, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        key: m!(HostString),
        query: m!(HostString),
    ) -> Result<FloatTensor<m!(HostFloat64Tensor), MirroredT>>
    where
        HostString: KnownType<S>,
        HostFloat32Tensor: KnownType<S>,
        HostFloat64Tensor: KnownType<S>,
        HostPlacement: PlacementLoad<S, m!(HostString), m!(HostString), m!(HostFloat64Tensor)>,
    {
        let z = plc.load(sess, &key, &query);
        Ok(FloatTensor::Host(z))
    }
}

impl SaveOp {
    pub(crate) fn float_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        key: m!(HostString),
        x: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<m!(HostUnit)>
    where
        HostString: KnownType<S>,
        HostUnit: KnownType<S>,
        HostPlacement: PlacementSave<S, m!(HostString), HostFloatT, m!(HostUnit)>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(_v) => unimplemented!(),
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
            FloatTensor::Mirrored3(_v) => unimplemented!(),
        };

        Ok(plc.shape(sess, &x))
    }
}

impl ConstantOp {
    pub(crate) fn float_kernel<S: Session, HostFloatT, MirroredT>(
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
    pub(crate) fn mir3_float_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &Mirrored3Placement,
        value: Constant,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementConstant<S, HostFloatT>,
        Mir3Tensor<HostFloatT>: Into<MirroredT>,
    {
        let (player0, player1, player2) = plc.host_placements();

        let z0 = player0.constant(sess, value.clone());
        let z1 = player1.constant(sess, value.clone());
        let z2 = player2.constant(sess, value);

        Ok(FloatTensor::Mirrored3(
            Mir3Tensor {
                values: [z0, z1, z2],
            }
            .into(),
        ))
    }
}

impl InputOp {
    pub(crate) fn float_kernel<S: Session, HostFloatT, MirroredT>(
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
    pub(crate) fn float_kernel<S: Session, HostFloatT, MirroredT>(
        sess: &S,
        plc: &HostPlacement,
        x: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementOutput<S, HostFloatT, HostFloatT>,
    {
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(_v) => unimplemented!(),
        };

        Ok(FloatTensor::Host(plc.output(sess, &x)))
    }
}

impl MuxOp {
    pub(crate) fn float_host_kernel<S: Session, HostFloatT, MirroredT, HostBitT, RepBitT>(
        sess: &S,
        plc: &HostPlacement,
        s: BoolTensor<HostBitT, RepBitT>,
        x: FloatTensor<HostFloatT, MirroredT>,
        y: FloatTensor<HostFloatT, MirroredT>,
    ) -> Result<FloatTensor<HostFloatT, MirroredT>>
    where
        HostPlacement: PlacementReveal<S, RepBitT, HostBitT>,
        HostPlacement: PlacementDemirror<S, MirroredT, HostFloatT>,
        HostPlacement: PlacementMux<S, HostBitT, HostFloatT, HostFloatT, HostFloatT>,
    {
        let s = match s {
            BoolTensor::Host(v) => v,
            BoolTensor::Replicated(v) => plc.reveal(sess, &v),
        };
        let x = match x {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(v) => plc.demirror(sess, &v),
        };
        let y = match y {
            FloatTensor::Host(v) => v,
            FloatTensor::Mirrored3(v) => plc.demirror(sess, &v),
        };

        let z = plc.mux(sess, &s, &x, &y);
        Ok(FloatTensor::Host(z))
    }
}
