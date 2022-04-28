use super::*;
use crate::computation::SqrtOp;
use crate::error::Result;
use crate::execution::Session;
use crate::fixedpoint::FixedpointTensor;

impl SqrtOp {
    pub(crate) fn rep_rep_kernel<S: Session, RepFixedT, MirFixedT>(
        sess: &S,
        rep: &ReplicatedPlacement,
        x: RepFixedT,
    ) -> Result<RepFixedT>
    where
        RepFixedT: FixedpointTensor,
        ReplicatedPlacement: ShapeFill<S, RepFixedT, Result = MirFixedT>,
        ReplicatedPlacement: PlacementMul<S, MirFixedT, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementTruncPr<S, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementPow2<S, RepFixedT, RepFixedT>,
        ReplicatedPlacement: PlacementLog2<S, RepFixedT, RepFixedT>,
    {
        // 2^(0.5 * log_2 x) = (2^log_2 x)^0.5 = x^0.5 = sqrt(x)

        let log2_x = rep.log2(sess, &x);

        let half = rep.shape_fill(
            sess,
            0.5_f64.as_fixedpoint(x.fractional_precision() as usize),
            &x,
        );
        let shifted_exponent = rep.mul(sess, &half, &log2_x);
        let exponent = rep.trunc_pr(sess, x.fractional_precision(), &shifted_exponent);
        Ok(rep.pow2(sess, &exponent))
    }
}
