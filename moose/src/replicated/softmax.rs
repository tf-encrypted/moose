use super::*;
use crate::computation::MaximumOp;
use crate::error::Result;
use macros::with_context;

impl MaximumOp {
    pub(crate) fn kernel<S: Session, RepRingT, RepBitT, MirRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        x: &[RepRingT],
    ) -> Result<RepRingT>
    where
        RepRingT: Clone,
        ReplicatedPlacement: PlacementLessThan<S, RepRingT, RepRingT, RepBitT>,
        ReplicatedPlacement: PlacementMul<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementRingInject<S, RepBitT, RepRingT>,
        ReplicatedPlacement: PlacementNeg<S, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementMaximum<S, RepRingT, RepRingT>,
        ReplicatedPlacement: PlacementAdd<S, RepRingT, RepRingT, RepRingT>,
        ReplicatedPlacement: ShapeFill<S, RepRingT, Result = MirRingT>,
        ReplicatedPlacement: PlacementSub<S, MirRingT, RepRingT, RepRingT>,
    {
        let n = x.len();
        if n == 0 {
            Err(Error::InvalidArgument(
                "maximum op needs a non-empty array of tensors".to_string(),
            ))
        } else if n == 1 {
            Ok(x[0].clone())
        } else {
            let chunk1 = &x[0..n / 2];
            let chunk2 = &x[n / 2..n];
            let max_chunk1 = plc.maximum(sess, chunk1);
            let max_chunk2 = plc.maximum(sess, chunk2);

            let lesser = plc.less(sess, &max_chunk1, &max_chunk2);

            let lesser_ring = plc.ring_inject(sess, 0, &lesser);
            let ones = plc.shape_fill(sess, Constant::Ring64(1), &lesser_ring);

            let expr = with_context!(
                plc,
                sess,
                lesser_ring * max_chunk2 + (ones - lesser_ring) * max_chunk1
            );
            Ok(expr)
        }
    }
}

impl SoftmaxOp {
    pub(crate) fn rep_fixed_kernel<S: Session, RepFixedT, ShapeT, RepRingT, RepBitT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        axis: Option<u32>,
        upmost_index: usize,
        x: RepFixedT,
    ) -> Result<RepFixedT>
    where
        RepFixedT: FixedpointTensor,
        ReplicatedPlacement: PlacementIndexAxis<S, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementSub<S, RepFixedT, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementMaximum<S, RepFixedT, RepFixedT>,
        RepFixedTensor<RepRingT>: Into<RepFixedT>,
        ReplicatedPlacement: PlacementFill<S, ShapeT, RepRingT>,
        ReplicatedPlacement: PlacementShape<S, RepFixedT, ShapeT>,
        ReplicatedPlacement: PlacementSub<S, RepFixedT, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementLessThan<S, RepFixedT, RepFixedT, RepBitT>,
        ReplicatedPlacement: PlacementRingInject<S, RepBitT, RepRingT>,
        ReplicatedPlacement: PlacementMux<S, RepRingT, RepFixedT, RepFixedT, RepFixedT>,
    {
        let xs: Vec<_> = (0..upmost_index)
            .map(|index| rep.index_axis(sess, axis.map(|a| a as usize).unwrap(), index, &x))
            .collect();

        let xmax = rep.maximum(sess, &xs);
        let e_x = rep.sub(sess, &x, &xmax);

        // input sanitization
        let zeros_fill = rep.fill(sess, 0_u8.into(), &rep.shape(sess, &e_x));
        let zeros_rep = RepFixedTensor {
            tensor: zeros_fill,
            integral_precision: x.integral_precision(),
            fractional_precision: x.fractional_precision(),
        }
        .into();

        let min_val = (-1_f64 * 2_f64.powf((x.integral_precision() - 1).into()).ln())
            .as_fixedpoint(x.fractional_precision() as usize);
        let min_val_fill = rep.fill(sess, min_val.into(), &rep.shape(sess, &x));
        let lower_limit = RepFixedTensor {
            tensor: min_val_fill,
            integral_precision: x.integral_precision(),
            fractional_precision: x.fractional_precision(),
        }
        .into();

        let max_diff = rep.sub(sess, &x, &xmax); // x - max(x)

        // max(x) - x < get_limit
        let lower = rep.less(sess, &max_diff, &lower_limit);
        let lower_ring = rep.ring_inject(sess, 0, &lower);
        let lower_wall = rep.mux(sess, &lower_ring, &e_x, &zeros_rep);

        Ok(lower_wall)
    }
}
