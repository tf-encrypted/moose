use super::*;
use crate::computation::{MaximumOp, ReplicatedPlacement};
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
