use super::*;
use crate::computation::ArgmaxOp;
use crate::error::Result;
use crate::execution::Session;
use crate::fixedpoint::FixedpointTensor;
use macros::with_context;

impl ArgmaxOp {
    pub(crate) fn rep_kernel<S: Session, RepRingT>(
        sess: &S,
        plc: &ReplicatedPlacement,
        axis: usize,
        upmost_index: usize,
        x: RepFixedTensor<RepRingT>,
    ) -> Result<m!(ReplicatedRing64Tensor)>
    where
        ReplicatedRing64Tensor: KnownType<S>,
    {
        unimplemented!()
        // TODO (Here we do simple argmax using mux)
    }
}

impl ReplicatedPlacement {
    pub(crate) fn tree_reduce_argmax<S, RepT>(
        &self,
        sess: &S,
        x: &[(RepT, RepT)],
        op: fn(&Self, &S, &(RepT, RepT), &(RepT, RepT)) -> (RepT, RepT),
    ) -> (RepT, RepT)
    where
        RepT: Clone,
    {
        let v_len = x.len();
        if v_len == 1 {
            (x[0].0.clone(), x[0].1.clone())
        } else {
            let chunk1 = &x[0..v_len / 2];
            let chunk2 = &x[v_len / 2..v_len];

            let op_res_chunk1 = self.tree_reduce(sess, chunk1, op);
            let op_res_chunk2 = self.tree_reduce(sess, chunk2, op);
            op(self, sess, &op_res_chunk1, &op_res_chunk2)
        }
    }
}
